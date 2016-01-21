-- 1) Install Torch:
--        See http://torch.ch/docs/getting-started.html#_
-- 2) Install necessary packages:
--        $ luarocks install loadcaffe
--        $ luarocks install hdf5
-- 3) Download a model:
--        $ wget http://places2.csail.mit.edu/models/vgg16_places2.tar.gz
--        $ tar xzvf vgg_places2.tar.gz
-- 4) Put images inside 'images' directory 
-- 5) Modify this script's options below
-- 6) Run it
--        $ th extract.lua
-- 7) Read it
--       In 'features', there will be one HDF5 file per image
--       with the dataset 'feat' containing the feature

-- some options
local image_dir = 'images'
local out_dir = 'features'
local prototxt = 'vgg16_places2/deploy.prototxt'
local caffemodel = 'vgg16_places2/vgg16_places2.caffemodel'
local layer_to_extract = 37
local batch_size = 2
local image_size = 224
local gpu_device = 1 
local mean_image = {105, 114, 116}

-- load dependencies
require 'cutorch'     -- CUDA tensors
require 'nn'          -- neural network package
require 'cudnn'       -- fast CUDA routines for neural networks
require 'loadcaffe'   -- loads models from Caffe
require 'paths'       -- utilities for reading directories 
require 'image'       -- reading/processing images
require 'hdf5'        -- writing hdf5 files
require 'xlua'        -- for progress bar

-- set GPU device
-- check which GPUs are free with 'nvidia-smi'
-- first GPU is #1, second is #2, ...
cutorch.setDevice(gpu_device)

-- loads model from caffe
local model = loadcaffe.load(prototxt, caffemodel, 'cudnn');

model:evaluate() -- turn on evaluation model (e.g., disable dropout)
model:cuda() -- ship model to GPU

print(model) -- visualizes the model
print('extracting layer ' .. layer_to_extract)

-- tensor to store RGB images on GPU 
local input_images = torch.CudaTensor(batch_size, 3, image_size, image_size)

-- keep track of the original filenames from input_images
-- original_filenames[i] is the filename for the image input_images[i]
local original_filenames = {}

-- current index into input_images
local counter = 1

-- read all *.jpg files in the 'image_dir', and store in the array 'filepaths'
local filepaths = {};
for f in paths.files(image_dir, '.jpg') do
  table.insert(filepaths, f)
end
print('found ' .. #filepaths .. ' images')

-- function to read image from disk, and do preprocessing
-- necessary for caffe models
function load_caffe_image(impath)
  local im = image.load(impath)                 -- read image
  im = image.scale(im, image_size, image_size)  -- resize image
  im = im * 255                                 -- change range to 0 and 255
  im = im:index(1,torch.LongTensor{3,2,1})      -- change BGR --> RGB

  -- subtract mean
  for i=1,3 do
    im[{ i, {}, {} }]:add(-mean_image[i])
  end

  return im
end

-- function to run feature extraction
function extract_feat()
  -- do forward pass of model on the images
  model:forward(input_images)

  -- read the activations from the requested layer
  local feat = model.modules[layer_to_extract].output

  -- ship activations back to host memory
  feat = feat:float()

  -- save feature for item in batch
  for i=1,counter-1 do
    local hdf5_file = hdf5.open(out_dir .. '/' .. original_filenames[i] .. '.h5', 'w')
    hdf5_file:write('feat', feat[i])
    hdf5_file:close()
  end
end

-- loop over each image
for image_id, filepath in ipairs(filepaths) do
  xlua.progress(image_id, #filepaths) -- display progress

  -- read image and store on GPU
  input_images[counter] = load_caffe_image(image_dir .. '/' .. filepath)

  -- keep track of original filename
  original_filenames[counter] = filepath

  -- once we fill up the batch, extract, and reset counter
  counter = counter + 1
  if counter > batch_size then
    extract_feat()      -- extract
    counter = 1         -- reset counter
    input_images:zero() -- for sanity, zero images
  end
end

-- one last time for end of batch
if counter > 1 then
  extract_feat()
end
