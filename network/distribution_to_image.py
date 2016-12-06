def get_colorized_image( image, prediction ):
  T = 0.38
  epsilon = 1e-8

  annealed_mean = np.exp( np.log(prediction + epsilon) / T ) 
  annealed_mean /= np.sum(annealed_mean, axis = 2).reshape((256,256,1))
  
  predicted_coloring = np.dot(annealed_mean, quantized_array)
  colorized_image = np.zeros( (256,256,3) )
  colorized_image[:,:,0:1] = image
  colorized_image[:,:,1:] = predicted_coloring
  
  return Image.fromarray( (255 * color.lab2rgb(colorized_image)).astype(np.uint8) )