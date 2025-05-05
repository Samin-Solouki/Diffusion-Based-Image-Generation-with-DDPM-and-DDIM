# Diffusion-Based-Image-Generation-with-DDPM-and-DDIM
Implemented a Denoising Diffusion Probabilistic Model (DDPM) and Deterministic DDIM
---
##  Project Level

Advanced Deep Learning / Generative Modeling

##  Project Highlights

* Implemented a **Denoising Diffusion Probabilistic Model (DDPM)** and **Deterministic DDIM**.
* Trained a noise prediction model using MSE loss over 70 epochs.
* Compared **image generation quality** and **speed** between DDPM and DDIM.
* Quantitatively evaluated models using **FID Score**.
* Visualized sample generations during training.

---

##  Project Description

This project explores image generation using diffusion models, specifically **DDPM** and **DDIM**. The DDPM framework involves gradually adding noise to images and then learning to reverse this process to generate new samples. DDIM offers a deterministic alternative to accelerate generation.

---

##  Objective

* Understand and implement forward and reverse diffusion processes.
* Train a model to denoise noisy images at different time steps.
* Compare DDPM and DDIM in terms of sample quality and generation time.
* Evaluate generation diversity and realism using **FID Score**.

---

##  Data Collection

The project uses the Sprites 16x16 Dataset from Kaggle, which contains thousands of 16x16 pixel sprite images in PNG format. These sprites represent various characters with diverse visual traits, making the dataset ideal for training and evaluating generative image models like DDPM and DDIM. 
https://www.kaggle.com/datasets/bajajganesh/sprites-16x16-dataset 
Images were loaded, resized (if needed), and normalized before being passed to the diffusion pipeline.


---

##  Preprocessing & Feature Extraction

* Normalized image pixel values.
* Resized input images to a consistent shape.
* Calculated and cached beta values and derived parameters like alpha, alpha\_bar to optimize noise scheduling.

---

##  Model Architecture

* **U-Net**-like architecture with:

  * Time embedding layers to incorporate timestep information.
  * Convolutional blocks to process noisy images.
* Model predicts added Gaussian noise at each step.

---

##  Training & Evaluation

* Optimizer: Adam
* Learning Rate: 0.0001
* Epochs: 70
* Loss Function: **Mean Squared Error (MSE)**
* Dataset split into training and validation sets.
* Validation and training losses tracked and visualized over epochs.

**Training Behavior:**

* Training and validation loss both decreased steadily.
* Loss plateaued around epoch 60.

---

##  Sampling Implementation

* Implemented reverse process using predicted noise and precomputed schedule parameters.
* Added Gaussian noise in DDPM (stochastic generation).
* Set σₜ = 0 in DDIM for deterministic generation.

---

##  Results

###  Generated Samples

* Visual examples of samples generated at epochs 50, 60, and 70 showed progressive quality improvement.
* DDPM images had higher perceptual quality than DDIM.

*Generated images, epoch 70:
![image](https://github.com/user-attachments/assets/c417b927-523b-4cba-8944-38a0f481cc4e)
*Generated images, epoch 60:
![image](https://github.com/user-attachments/assets/1f3e32c9-1dcd-42f1-94ec-58fdd385333f)
*Generated images, epoch 50:
![image](https://github.com/user-attachments/assets/e6a30249-88d4-4a05-877c-fb760428c0fe)



###  Generation Time

* DDPM: 5m 12s
* DDIM: 5m 9s
  (Difference not substantial—likely affected by hardware and batch size.)

###  FID Scores

| Model | FID Score  |
| ----- | ---------- |
| DDPM  | **2.6855** |
| DDIM  | 9.8844     |

**Interpretation:**
DDPM outperformed DDIM in image quality, confirming that DDPM better captured the data distribution.

---

##  Conclusion

* DDPM offers superior image quality at the cost of slightly longer generation time.
* DDIM provides faster sampling but sacrifices some visual fidelity.
* The diffusion model successfully learned the data distribution, as evidenced by low FID scores.
* Precomputing schedule parameters and using time embeddings significantly helped training efficiency and effectiveness.

---

##  Future Work

* Implement **conditional diffusion** for class-based image generation.
* Explore **improved noise schedulers** (e.g., cosine schedule).
* Optimize inference with **fewer sampling steps** while maintaining image quality.

