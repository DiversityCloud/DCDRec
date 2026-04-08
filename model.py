import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionRecommender(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        x_dim: int = 64,
        item_embedding_dim: int = 64,
        num_diffusion_steps: int = 500,
        device: str = "cuda",
        guidance_scale: float = 0.9,
        unconditional_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.x_dim = x_dim
        self.item_embedding_dim = item_embedding_dim
        self.device = device
        self.guidance_scale = guidance_scale
        self.unconditional_prob = unconditional_prob

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.time_embedding = nn.Embedding(100, embedding_dim)

        self.seq_attention = nn.MultiheadAttention(embedding_dim, num_heads=4)
        self.user_attention = nn.MultiheadAttention(embedding_dim, num_heads=4)
        self.item_attention = nn.MultiheadAttention(embedding_dim, num_heads=4)

        self.fc = nn.Sequential(
            nn.Linear(x_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_items),
        )
        self.criterion = nn.CrossEntropyLoss()

        self.order_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.25]), requires_grad=True)
        self.combined_rep_mapper = nn.Linear(embedding_dim * 2, x_dim)

        self.num_diffusion_steps = num_diffusion_steps
        self.register_buffer("beta", self.linear_beta_schedule(num_diffusion_steps))
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_hat", torch.cumprod(self.alpha, dim=0))

        self.denoiser = nn.Sequential(
            nn.Linear(x_dim + 32 + x_dim, 256),
            nn.ReLU(),
            nn.Linear(256, x_dim),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

    @staticmethod
    def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
        return torch.linspace(beta_start, beta_end, T)

    def forward_diffusion(self, combined_rep: torch.Tensor, t: torch.Tensor):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).unsqueeze(1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).unsqueeze(1)
        epsilon = torch.randn_like(combined_rep)
        x_t = sqrt_alpha_hat * combined_rep + sqrt_one_minus_alpha_hat * epsilon
        return x_t, epsilon

    def reverse_diffusion_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_ave: torch.Tensor,
        time_embed: torch.Tensor,
        is_guided=True,
    ) -> torch.Tensor:
        if isinstance(is_guided, torch.Tensor):
            epsilon_cond = self.denoiser(torch.cat([z_ave, time_embed, x_t], dim=1))
            z_uncond = torch.zeros_like(z_ave)
            epsilon_uncond = self.denoiser(torch.cat([z_uncond, time_embed, x_t], dim=1))
            epsilon = epsilon_uncond + self.guidance_scale * (
                epsilon_cond - epsilon_uncond
            ) * is_guided.float().unsqueeze(1)
        else:
            if is_guided:
                epsilon = self.denoiser(torch.cat([z_ave, time_embed, x_t], dim=1))
            else:
                z_uncond = torch.zeros_like(z_ave)
                epsilon = self.denoiser(torch.cat([z_uncond, time_embed, x_t], dim=1))
        return epsilon

    def generate_condition(self, user_denoised_output: torch.Tensor, item_denoised_output: torch.Tensor) -> torch.Tensor:
        combined_representation = torch.cat([user_denoised_output, item_denoised_output], dim=1)
        return self.combined_rep_mapper(combined_representation)

    def safe_mean(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        if tensor.size(dim) == 0:
            return torch.zeros(tensor.size(0), self.embedding_dim, device=tensor.device)
        return tensor.mean(dim=dim)

    def encode_context(self, batch: dict):
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        user_ids = batch["user_ids"]
        movie_sequences = batch["movie_sequences"]
        time_sequences = batch["time_sequences"]
        first_order_users = batch["first_order_users"]
        first_order_movies = batch["first_order_movies"]
        second_order_movies = batch["second_order_movies"]
        third_order_movies = batch["third_order_movies"]

        user_emb = self.user_embedding(user_ids)
        movie_seq_emb = self.item_embedding(movie_sequences)

        time_seq_emb = []
        for i in range(6):
            time_feature = time_sequences[:, :, i]
            time_emb = self.time_embedding(time_feature)
            time_seq_emb.append(time_emb)
        time_seq_emb = torch.stack(time_seq_emb).sum(dim=0)
        seq_emb = movie_seq_emb + time_seq_emb

        q_seq = seq_emb.permute(1, 0, 2)
        seq_selfatt_output, _ = self.seq_attention(q_seq, q_seq, q_seq)
        seq_selfatt_output = seq_selfatt_output.permute(1, 0, 2)

        user_neighbors_emb = self.user_embedding(first_order_users).mean(dim=1)
        first_order_item_emb = self.safe_mean(self.item_embedding(first_order_movies), dim=1)
        second_order_item_emb = self.safe_mean(self.item_embedding(second_order_movies), dim=1)
        third_order_item_emb = self.safe_mean(self.item_embedding(third_order_movies), dim=1)

        order_weights = F.softmax(self.order_weights, dim=0)
        item_neighbors_emb = (
            order_weights[0] * first_order_item_emb
            + order_weights[1] * second_order_item_emb
            + order_weights[2] * third_order_item_emb
        )

        q_user = seq_selfatt_output.permute(1, 0, 2)
        kv_user = user_neighbors_emb.unsqueeze(0)
        q_item = seq_selfatt_output.permute(1, 0, 2)
        kv_item = item_neighbors_emb.unsqueeze(0)

        user_denoised_output, _ = self.user_attention(q_user, kv_user, kv_user)
        item_denoised_output, _ = self.item_attention(q_item, kv_item, kv_item)

        user_denoised_output = user_denoised_output.mean(dim=0)
        item_denoised_output = item_denoised_output.mean(dim=0)
        z_ave = self.generate_condition(user_denoised_output, item_denoised_output)

        user_denoised_norm = F.normalize(user_denoised_output, dim=1)
        item_denoised_norm = F.normalize(item_denoised_output, dim=1)
        sim_matrix = torch.matmul(user_denoised_norm, item_denoised_norm.T)
        labels = torch.arange(user_ids.size(0), device=self.device)
        contrastive_loss = F.cross_entropy(sim_matrix / 0.1, labels)

        combined_representation = torch.cat([user_denoised_output, item_denoised_output], dim=1)
        combined_rep = self.combined_rep_mapper(combined_representation)
        logits = self.fc(combined_rep)

        return batch, logits, contrastive_loss, z_ave, combined_rep, user_emb, item_neighbors_emb

    def forward(self, batch: dict, training: bool = True):
        batch, logits, contrastive_loss, z_ave, combined_rep, user_emb, item_neighbors_emb = self.encode_context(batch)
        targets = batch["target_movies"]
        batch_size = targets.size(0)

        if training:
            t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device).long()
            diffusion_loss = self.calculate_diffusion_loss(combined_rep, z_ave, t)
            total_loss = self.calculate_loss(logits, targets, combined_rep, contrastive_loss, diffusion_loss)
            return {
                "logits": logits,
                "targets": targets,
                "contrastive_loss": contrastive_loss,
                "diffusion_loss": diffusion_loss,
                "total_loss": total_loss,
                "z_ave": z_ave,
                "combined_rep": combined_rep,
                "user_emb": user_emb,
                "item_neighbors_emb": item_neighbors_emb,
            }

        x_t = torch.randn(batch_size, self.x_dim, device=self.device)
        for step in reversed(range(self.num_diffusion_steps)):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            t_embed = t.float().unsqueeze(1)
            time_embed = self.time_mlp(t_embed)
            is_guided = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            epsilon = self.reverse_diffusion_step(x_t, t, z_ave, time_embed, is_guided=is_guided)

            alpha_t = self.alpha[t].unsqueeze(1)
            alpha_hat_t = self.alpha_hat[t].unsqueeze(1)
            sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
            coef = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
            x_prev = sqrt_recip_alpha * (x_t - coef * epsilon)

            if step > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(self.beta[t]).unsqueeze(1)
                x_prev = x_prev + sigma_t * noise
            x_t = x_prev

        combined_rep_recovered = x_t
        logits_recovered = self.fc(combined_rep_recovered)
        return logits_recovered

    def calculate_diffusion_loss(self, combined_rep: torch.Tensor, z_ave: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_t, epsilon = self.forward_diffusion(combined_rep, t)
        t_embed = t.float().unsqueeze(1)
        time_embed = self.time_mlp(t_embed)

        is_unconditional = torch.rand(t.size(0), device=self.device) < self.unconditional_prob
        is_guided = ~is_unconditional
        z_ave_cond = z_ave.clone()
        z_ave_cond[is_unconditional] = 0

        predicted_epsilon = self.reverse_diffusion_step(x_t, t, z_ave_cond, time_embed, is_guided=is_guided)
        return F.mse_loss(predicted_epsilon, epsilon)

    def calculate_loss(
        self,
        logits: torch.Tensor,
        target_movies: torch.Tensor,
        combined_rep: torch.Tensor,
        contrastive_loss: torch.Tensor,
        diffusion_loss: torch.Tensor,
        lambda_contrastive: float = 0.1,
        lambda_diffusion: float = 1.0,
        alpha_embed: float = 0.01,
    ) -> torch.Tensor:
        classification_loss = self.criterion(logits, target_movies)
        target_item_embed = self.item_embedding(target_movies)
        loss_embed = F.mse_loss(combined_rep, target_item_embed)
        total_loss = (
            classification_loss
            + alpha_embed * loss_embed
            + lambda_contrastive * contrastive_loss
            + lambda_diffusion * diffusion_loss
        )
        return total_loss
