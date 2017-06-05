# Image-Processing---Tensorflow

Change directory to one where working_directory folder is created, please see below for example. Here working_directory folder is created on path ~/Python/Python35/Lib/site-packages/

cd ~/Python/Python35/Lib/site-packages/

# Command to call training script
python train_my_model.py \
--bottleneck_dir= working_directory/bottlenecks/ \
--how_many_training_steps=500 \
--model_dir=working_directory/inception \
--output_graph=working_directory/retrained_graph.pb \
--output_labels=working_directory/retrained_labels.txt \
--image_dir=working_directory/images/\

# Command to call classification script
python classify_image.py ~/Python/Python35/Lib/site-packages/ ~/Python/Python35/Lib/site-packages/working_directory/test_image/ ~/Python/Python35/Lib/site-packages/working_directory/cat_images/

~/Python/Python35/Lib/site-packages/ -- a directory where working_directory folder is created

~/Python/Python35/Lib/site-packages/working_directory/test_image/  -- folder which contains images for classification

~/Python/Python35/Lib/site-packages/working_directory/cat_images/ -- folder where cat images will be moved after classification. Kindly note that if an image with same name already exists in cat_images folder, that image will be over written.
