Diffusion Model From Scratch

A class-conditional image generation model (diffusion model) built entirely from scratch in PyTorch — no pretrained models, no external APIs. Trained on CPU on a personal laptop.

This project implements the same core technique behind tools like Stable Diffusion and DALL-E, at a small scale: a model learns to reverse a noise-adding process, allowing it to generate new images from pure random noise.

What it does
Generates 28x28 grayscale images of clothing items (Fashion-MNIST categories: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
Supports class-conditional generation — request a specific category and the model generates it
Includes an embedding-blending experiment that interpolates between two categories, showing the model learned a meaningful, continuous representation of clothing shapes rather than 10 disconnected labels
How it works
Forward process (step1_forward_process.py): implements the noise schedule and visualizes an image gradually turning into random noise over 200 steps.
U-Net training (step2_train_unet.py): a U-Net with skip connections is trained to predict the noise added to an image at a given timestep, using MSE loss.
Reverse sampling / generation (step3_generate.py): starting from pure noise, the trained model repeatedly predicts and removes noise to produce a new image.
Class-conditional model (step4_train_conditional.py, step5_generate_conditional.py): extends the U-Net to take a class label (via a learned embedding) alongside the timestep, allowing generation of a specific requested category.
Embedding interpolation (step6_blend_categories.py): blends two categories' learned embeddings and generates images along the blend, exploring the model's learned representation space.
Technical highlights
Implemented the full DDPM (Denoising Diffusion Probabilistic Models) math from scratch: noise schedules, closed-form forward noising, and the reverse sampling update rule.
Built a U-Net with skip connections and timestep/class conditioning via learned embeddings.
Diagnosed and fixed a real training bug: resuming training with a freshly-initialized optimizer (rather than restoring saved optimizer state) caused a temporary but significant quality regression. Fixed by implementing proper checkpointing — saving and restoring both model weights and optimizer state — enabling stable training across multiple sessions.
Built a fairer evaluation approach: since diffusion sampling is stochastic, single-sample comparisons are misleading. Generation script produces multiple samples per category in a grid to assess consistency rather than one-off luck.
Tech stack
Python, PyTorch (CPU), NumPy, Matplotlib, Torchvision (Fashion-MNIST dataset)
Notes
Trained entirely on a laptop CPU (no GPU) — training scripts include full progress logging.
Dataset auto-downloads via torchvision.datasets.FashionMNIST on first run.
Trained model weights are not included in this repo (regenerate by running step4_train_conditional.py).
