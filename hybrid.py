class HybridResNetTransformer(nn.Module):
    """ResNet101 → CBAM → Patch embedding → Transformer → MLP head."""
    def __init__(self, num_classes=NUM_CLASSES, resnet_pretrained=True, embed_dim=256, transformer_depth=4,
                 transformer_heads=8, transformer_mlp_dim=1024, transformer_dropout=0.5, head_dropout=0.5,
                 use_cls_token=True, freeze_backbone=False):
        super().__init__()
        self.use_cls_token = use_cls_token
        try:
            resnet = (models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if resnet_pretrained else None))
        except AttributeError:
            resnet = (models.resnet101(pretrained=resnet_pretrained))
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        backbone_out_channels = 2048
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.cbam = CBAM(backbone_out_channels)
        self.patch_embed = PatchEmbeddingFromFeatureMap(
            in_channels=backbone_out_channels, embed_dim=embed_dim,
            spatial_size=(7, 7), use_cls_token=use_cls_token
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=transformer_heads, dim_feedforward=transformer_mlp_dim,
            dropout=transformer_dropout, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(512, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.cbam(feat)
        tokens = self.patch_embed(feat)
        tokens = self.transformer(tokens)
        cls_rep = tokens[:, 0] if self.use_cls_token else tokens.mean(dim=1)
        return self.mlp_head(cls_rep)

