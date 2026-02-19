# Patch Embedding (project feature map -> tokens)
class PatchEmbeddingFromFeatureMap(nn.Module):

    def __init__(self, in_channels, embed_dim, use_cls_token=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.use_cls_token = use_cls_token
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed = None  

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2)  
        N = H * W

        if self.pos_embed is None or self.pos_embed.shape[1] != (N + (1 if self.use_cls_token else 0)):
            
            nb_tokens = N + (1 if self.use_cls_token else 0)
            self.pos_embed = nn.Parameter(torch.randn(1, nb_tokens, self.embed_dim))

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  
            x = torch.cat([cls_tokens, x], dim=1)  
      
        x = x + self.pos_embed
        return x 

