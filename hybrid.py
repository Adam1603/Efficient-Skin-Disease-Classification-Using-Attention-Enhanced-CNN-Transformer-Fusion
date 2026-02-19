# Hybrid Model: ResNet101 -> CBAM -> PatchEmbedding -> Transformer -> MLP Head

class HybridResNetTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        resnet_pretrained=True,
        embed_dim=768,
        transformer_depth=8,
        transformer_heads=8,
        transformer_mlp_dim=2048,
        transformer_dropout=0.5,
        use_cls_token=True,
        freeze_backbone=False
    ):
        super().__init__()

    
        resnet = models.resnet101(pretrained=resnet_pretrained)
 
        self.backbone = nn.Sequential(
            resnet.conv1,  
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4  
        )
        backbone_out_channels = 2048

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.cbam = CBAM(backbone_out_channels, reduction=16)

        self.patch_embed = PatchEmbeddingFromFeatureMap(in_channels=backbone_out_channels, embed_dim=embed_dim, use_cls_token=use_cls_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_mlp_dim,
            dropout=transformer_dropout,
            activation='relu',
            batch_first=True  
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

     
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_classes)
            )
        else:
                self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_classes)
            )

    
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        feat = self.backbone(x)  
        feat = self.cbam(feat)   
        tokens = self.patch_embed(feat)  
        tokens = self.transformer(tokens)
        if self.use_cls_token:
            cls_rep = tokens[:, 0] 
            out = self.mlp_head(cls_rep)  
        else:

            pooled = tokens.mean(dim=1)  
            out = self.mlp_head(pooled)

        return out


