```mermaid
graph TD;
    A[Input: Images (6 channels)] --> B[Image Processing CNN];
    B --> B1[Conv2D (6->32) + BatchNorm + ReLU + MaxPool];
    B1 --> B2[Conv2D (32->64) + BatchNorm + ReLU + MaxPool];
    B2 --> B3[Conv2D (64->128) + BatchNorm + ReLU + MaxPool];
    B3 --> B4[Flatten + Fully Connected (128x28x28 -> 512)];
    B4 --> B5[Dropout];

    A2[Input: Color Data (RGB)] --> C[Color Fully Connected Network];
    C --> C1[Linear (3 -> 64)];
    C1 --> C2[ReLU + Dropout];

    B5 --> D[Feature Concatenation];
    C2 --> D;

    D --> E[Combined Classifier];
    E --> E1[Linear (512+64 -> 128)];
    E1 --> E2[ReLU + Dropout];
    E2 --> E3[Linear (128 -> num_classes)];
    E3 --> F[Output: Class Scores];
```