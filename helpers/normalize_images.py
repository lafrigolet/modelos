import torchvision.transforms.functional as TF


def normalize_image(img, target_width, target_height):
    # Get the width and height of the image
    width, height = img.size
    tw, th = target_width, target_height
    
    # Calculate the new width and height while keeping aspect ratio
    if tw/width > th/height:
        new_width  = int(th * width / height)
        new_height = th
    else:
        new_width  = tw
        new_height = int(tw * height / width)
        
    #plt.imshow(img, cmap="gray"); plt.show();
        
    # Convert the image to grayscale
    img_gray = TF.to_grayscale(img, num_output_channels=1)
    #plt.imshow(img_gray, cmap="gray"); plt.show();
    
    #img_crop = TF.crop(img_gray, 0, 0, 50, 150)
    #plt.imshow(img_crop, cmap="gray"); plt.show();
    
    # Resize the image while keeping aspect ratio
    img_resized = TF.resize(img_gray, (new_height, new_width))
    #plt.imshow(img_resized, cmap="gray"); plt.show();
    
    # Pad the image if necessary to match the target size
    img_padded = TF.pad(img_resized, (0, 0, tw - new_width, th - new_height))
    #img_padded = img_padded / 255
    # plt.imshow(img_padded, cmap="gray"); plt.show();
    
    # Convert the image to a PyTorch tensor 
    # (height, width, channel) -> (channel, height, width)
    img_tensor = TF.to_tensor(img_padded)
    
    return img_tensor
