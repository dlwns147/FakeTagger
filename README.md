
 This repository is a simple, unofficial implementation of "FakeTagger: Robust Safeguards against DeepFake Dissemination via Provenance Tracking" (https://arxiv.org/abs/2009.09869).

# Simple FakeTagger implementation

![image](https://user-images.githubusercontent.com/77950714/173223624-f2afc2b4-500f-4ee2-81f2-11a7b1c0b4f4.png)

 FakeTagger is composed of 5 modules.
 
 1. **Message Generator** creates a message with 0, and 1 and adds redundant data on the message.
 2. **Image Encoder** inserts the redundant message generated by the Message Generator into the original image, thereby generating an *embedded* image that is indistinguishable from the original input image.
 3. **GAN Simulator** applies a deepfake algorithm to the *embedded* image, generating a *manipulated embedded* image.
 4. **The image decoder** extracts redundant messages from the embedded image or the *manipulated embedded* image.
 5. **Message Decoder** extracts the original message from the redundant message.

 In this repository, only Image Encoder, GAN Simulator, Image Decoder are implemented.
 
 ### Image Encoder, Decoder
 ![image](https://user-images.githubusercontent.com/77950714/173226563-75952226-d58b-4320-b3a8-b49b37f9af2b.png)

  - Image encoder combines a message and an image into an embedded image. We utilize U-Net same as the original paper. The message is concatenated with the output of the U-Net encoder and then passed to the U-Net decoder.
  
  - U-Net source code : https://colab.research.google.com/github/usuyama/pytorch-unet/blob/master/pytorch_unet_resnet18_colab.ipynb
  
  - Image decoder extracts a message from an input image. This is composed of 7 convolution layers following an FC layer and a sigmoid layer, generating a vector of which the length is the same as the message.
 
 ![image](https://user-images.githubusercontent.com/77950714/173226674-ac13c08f-9a8a-416c-8862-282524e358f2.png)

### GAN Simulator
-  GAN Simulator manipulates an embedded image with a deepfake algorithm. We utilize **faceswap** algorithm which changes the faces of two people with one encoder and two decoders.
-  Faceswap source : https://github.com/Oldpan/Faceswap-Deepfake-Pytorch

### Training

```
python train.py --batch_size 128 --name name
```

### Inference

```
python train.py --batch_size 128 --name name
```
 
 ### Examples of Generated Images
 
 ![image](https://user-images.githubusercontent.com/77950714/173226638-e1ee9f6b-d5df-4a46-9b77-138a7d05b3ce.png)

![image](https://user-images.githubusercontent.com/77950714/173226646-2e10034c-a593-42a9-ae54-ff5c4cb1847e.png)


 ### Experimental Results
 
 #### 1) 𝜆 = 1, 𝛼 = 0.1
 ![image](https://user-images.githubusercontent.com/77950714/173226808-37daa08a-b233-49ce-aeb2-34a07351e9a4.png)

 This table shows how exactly the image decoder extracts messages from embedded images and manipulated embedded images. In most cases, the image decoder obtains the messages from input images. Only the accuracy of the manipulated embedded images decreases by 4% points from the embedded images.

 ![image](https://user-images.githubusercontent.com/77950714/173226813-52a25ecc-56ca-402b-8174-058f1ed1a611.png)

 This table shows how simmilar the embedded images and the original images are each other in terms of PSNR (Peak Signal-to-noise ratio) and SSIM (Structural Similarity Index Measure).

 #### 2) 𝜆 = 10, 𝛼 = 0.1
 ![image](https://user-images.githubusercontent.com/77950714/173226839-f26c00ce-b347-4392-adac-67fdab1ba600.png)

![image](https://user-images.githubusercontent.com/77950714/173226843-dc7677b7-264e-4cd0-9a75-571ba746ff56.png)

  The image encoder generates easily-distinguishable images when 𝜆 is too large. The above figure and table show the input images/output images of each module and the message reconstruction accuracy when 𝜆 = 10, 𝛼 = 0.1. You can see that the message reconstruction accuracy increases while the similarity of images decreases when 𝜆 increases.


 #### 3) 𝜆 = 1, 𝛼 = 0.5
 ![image](https://user-images.githubusercontent.com/77950714/173226866-6539001a-ee3a-4553-afa4-d227ebdc4e50.png)
 ![image](https://user-images.githubusercontent.com/77950714/173226869-7ade5cc2-2480-46e6-a4bb-6b3915197771.png)
 
 If 𝛼 is too large, the image decoder can't extract the messages from input images. The above figure and table show the input images/output images of each module and the message reconstruction accuracy when 𝜆 = 1, 𝛼 = 0.5. You can see that the image similarity increases while the message reconstruction accuracy decreases when 𝛼 increases.
