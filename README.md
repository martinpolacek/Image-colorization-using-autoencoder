 Coloring black and white images using neural network

This project deals with the topic of colouring black and white images using a neural network based on the principle of autoencoder. 

## Dataset
I used a total of 75,000 images as the dataset, which came from the following categories:
- Portraits (women, men, children, actors)
- Landscapes (forests, beaches, mountains, flowers, waterfalls, islands)
- Animals (zoo, savannah)
- Cities (buildings, cars, streets)

I chose vertical mirroring of the images as an augmentation, which expanded the dataset to a size of 150,000 images.

![Mirror Augmentation](README_data/augmentation.png?raw=true "Mirror Augmentation")

Due to computational complexity, I chose images of 255 x 255 pixels. 

## Color model
For the color model, I used the Lab model, which contains a luminance component in one channel, so the neural network can only estimate the two channels (a,b) that make up the color component of the image.

## Network model
As a neural network model, I chose a network that contains only convolutional and deconvolutional layers. A significant improvement in detail was obtained by connecting the output of the pre-trained Resnet network, which serves as an image classifier.

Overall, the network contains 8,369,615 trainable parameters. The resnet parameters are locked for training.

![Model](README_data/resnet%20%2B%20autoenkoder.png?raw=true "Network structure")

## Results

The results depend largely on the input data. There is a large range of image types, especially for animals and portraits. To work properly in all categories, the dataset needs to be expanded by at least 200,000 images.

Image comparison (black and white, color, original, channel a):

![Model](README_data/finalni_correct_porovnani.jpg "Network structure")

## Files

The attached files contain an already trained model and a training script that allows training on multiple graphics cards.

Training on 8 GPUs with 150000 images of 255x255 took 12 hours.
