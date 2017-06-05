import tensorflow as tf
import sys
import os
import glob
import shutil

root_dir = sys.argv[1]
src_dir = sys.argv[2]
dst_dir = sys.argv[3]



with tf.Session() as sess:
	
	for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
	
		
		image_path = jpgfile
					
		image_data = tf.gfile.FastGFile(image_path, 'rb').read()
				
		label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(root_dir + "/tf_files/retrained_labels.txt")]
		
		with tf.gfile.FastGFile(root_dir + "/tf_files/retrained_graph.pb", 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')
			    
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
		predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    		
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
		print(jpgfile)
		for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			print('%s (score = %.5f)' % (human_string, score))
			if human_string == 'cat':
				cat_score = score
		
		if cat_score > 0.7:			
			shutil.copy(jpgfile, dst_dir)	
