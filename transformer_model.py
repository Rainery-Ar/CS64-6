# transformer_model.py
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Net(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_chans=1, num_classes=10,
                 embed_dim=128, depth=1, num_heads=4, mlp_ratio=2.0, p=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=p, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                  # (B, 16, C)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, C)
        x = torch.cat([cls, x], dim=1)           # (B, 17, C)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        x = self.encoder(x)                      # (B, 17, C)
        cls_out = self.norm(x[:, 0])
        logits = self.head(cls_out)              # (B, 10)
        return logits
