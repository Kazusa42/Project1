ConvNeXt(
  (downsample_layers): ModuleList(
    (0): Sequential(
      (0): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
      (1): LayerNorm()
    )
    (1): Sequential(
      (0): LayerNorm()
      (1): Conv2d(96, 192, kernel_size=(2, 2), stride=(2, 2))
    )
    (2): Sequential(
      (0): LayerNorm()
      (1): Conv2d(192, 384, kernel_size=(2, 2), stride=(2, 2))
    )
    (3): Sequential(
      (0): LayerNorm()
      (1): Conv2d(384, 768, kernel_size=(2, 2), stride=(2, 2))
    )
  )
  (stages): ModuleList(
    (0): Sequential(
      (0): Block(
        (dwconv): Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=96, out_features=384, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=384, out_features=96, bias=True)
        (drop_path): Identity()
      )
      (1): Block(
        (dwconv): Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=96, out_features=384, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=384, out_features=96, bias=True)
        (drop_path): Identity()
      )
      (2): Block(
        (dwconv): Conv2d(96, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=96)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=96, out_features=384, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=384, out_features=96, bias=True)
        (drop_path): Identity()
      )
    )
    (1): Sequential(
      (0): Block(
        (dwconv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=192, out_features=768, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=768, out_features=192, bias=True)
        (drop_path): Identity()
      )
      (1): Block(
        (dwconv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=192, out_features=768, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=768, out_features=192, bias=True)
        (drop_path): Identity()
      )
      (2): Block(
        (dwconv): Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=192, out_features=768, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=768, out_features=192, bias=True)
        (drop_path): Identity()
      )
    )
    (2): Sequential(
      (0): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
        (drop_path): Identity()
      )
      (1): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
        (drop_path): Identity()
      )
      (2): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
        (drop_path): Identity()
      )
      (3): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
        (drop_path): Identity()
      )
      (4): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
        (drop_path): Identity()
      )
      (5): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
        (drop_path): Identity()
      )
      (6): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
        (drop_path): Identity()
      )
      (7): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
        (drop_path): Identity()
      )
      (8): Block(
        (dwconv): Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=384, out_features=1536, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=1536, out_features=384, bias=True)
        (drop_path): Identity()
      )
    )
    (3): Sequential(
      (0): Block(
        (dwconv): TransNeck(
          (conv1): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Sequential(
            (0): MHSA(
              (query): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
              (key): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
              (value): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
              (softmax): Softmax(dim=-1)
            )
          )
          (bn2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
        (drop_path): Identity()
      )
      (1): Block(
        (dwconv): TransNeck(
          (conv1): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Sequential(
            (0): MHSA(
              (query): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
              (key): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
              (value): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
              (softmax): Softmax(dim=-1)
            )
          )
          (bn2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
        (drop_path): Identity()
      )
      (2): Block(
        (dwconv): TransNeck(
          (conv1): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Sequential(
            (0): MHSA(
              (query): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
              (key): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
              (value): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1))
              (softmax): Softmax(dim=-1)
            )
          )
          (bn2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (shortcut): Sequential()
        )
        (norm): LayerNorm()
        (pwconv1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU()
        (pwconv2): Linear(in_features=3072, out_features=768, bias=True)
        (drop_path): Identity()
      )
    )
  )
  (norm0): LayerNorm()
  (norm1): LayerNorm()
  (norm2): LayerNorm()
  (norm3): LayerNorm()
)

