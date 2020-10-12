import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
import inputs as data
import threading

## Iteration
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    if shuffle:
      excerpt = indices[start_idx:start_idx + batch_size]
    else:
      excerpt = slice(start_idx, start_idx + batch_size)
    yield inputs[excerpt], targets[excerpt]

## Threading
def threading_data(data=None, fn=None, **kwargs):
  # define function for threading
  def apply_fn(results, i, data, kwargs):
    results[i] = fn(data, **kwargs)

  ## start multi-threaded reading.
  results = [None] * len(data) ## preallocate result list   构造result
  threads = []    #线程的列表
  for i in range(len(data)):
    t = threading.Thread(
                    name='threading_and_return',
                    target=apply_fn,
                    args=(results, i, data[i], kwargs)
                    )              #创建线程执行target函数即fn去修改result
    t.start()
    threads.append(t)
  ## <Milo> wait for all threads to complete 等待所有线程结束
  for t in threads:
    t.join()   
  return np.asarray(results)       
## isoTrainImageGenerator
def isoTrainImageGenerator(filepath, batch_size, seq_len, num_classes, modality):
  X_train,y_train = data.load_iso_video_list(filepath)      #读取文件的路径以及label
  X_tridx = np.asarray(np.arange(0, len(y_train)), dtype=np.int32)    #创建顺序的array
  y_train = np.asarray(y_train, dtype=np.int32)  
  while 1:
    for X_indices, y_label_t in minibatches(X_tridx, y_train, 
                                            batch_size, shuffle=True):  #将数据打乱顺序取出一组，每一组大小为batch_size这么大
      # Read data for each batch      
      image_path = []   #保存路径
      image_fcnt = []   #保存framecnt帧数？
      image_olen = []   #一个batch取的图像数目  
      image_start = []  #开始的帧数
      is_training = []  #是否为训练
      for data_a in range(batch_size):   
        X_index_a = X_indices[data_a]    #取出这个index
        key_str = '%06d' % X_index_a     
        image_path.append(X_train[key_str]['videopath'])   #找到这个index对应图像文件的路径
        image_fcnt.append(X_train[key_str]['framecnt'])    #对应的framecnt存入相应的数组
        image_olen.append(seq_len)
        image_start.append(1)
        is_training.append(True) # Training
      image_info = zip(image_path,image_fcnt,image_olen,image_start,is_training)    #数组打包为一个新的大数组
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_rgb_data)    #传入data与函数
      elif modality==1: #Depth
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_depth_data)
      elif modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_flow_data)     
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)
  
## isoTestImageGenerator
def isoTestImageGenerator(filepath, batch_size, seq_len, num_classes, modality):
  X_test,y_test = data.load_iso_video_list(filepath)
  X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
  y_test  = np.asarray(y_test, dtype=np.int32)
  while 1:
    for X_indices, y_label_t in minibatches(X_teidx, y_test, 
                                            batch_size, shuffle=False):
      # Read data for each batch      
      image_path = []
      image_fcnt = []
      image_olen = []
      image_start = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_test[key_str]['videopath'])
        image_fcnt.append(X_test[key_str]['framecnt'])
        image_olen.append(seq_len)
        image_start.append(1)
        is_training.append(False) # Testing
      image_info = zip(image_path,image_fcnt,image_olen,image_start,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_rgb_data)
      elif modality==1: #Depth
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_depth_data)
      elif modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_flow_data)     
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)

## jesterTrainImageGenerator
def jesterTrainImageGenerator(filepath, batch_size, seq_len, num_classes, modality):
  X_train,y_train = data.load_iso_video_list(filepath)
  X_tridx = np.asarray(np.arange(0, len(y_train)), dtype=np.int32)
  y_train = np.asarray(y_train, dtype=np.int32)
  while 1:
    for X_indices, y_label_t in minibatches(X_tridx, y_train, 
                                            batch_size, shuffle=True):
      # Read data for each batch      
      image_path = []
      image_fcnt = []
      image_olen = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_train[key_str]['videopath'])
        image_fcnt.append(X_train[key_str]['framecnt'])
        image_olen.append(seq_len)
        is_training.append(True) # Training
      image_info = zip(image_path,image_fcnt,image_olen,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_rgb_data)
      if modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_flow_data)
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)
  
## jesterTestImageGenerator
def jesterTestImageGenerator(filepath, batch_size, seq_len, num_classes, modality):
  X_test,y_test = data.load_iso_video_list(filepath)
  X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
  y_test  = np.asarray(y_test, dtype=np.int32)
  while 1:
    for X_indices, y_label_t in minibatches(X_teidx, y_test, 
                                            batch_size, shuffle=False):
      # Read data for each batch      
      image_path = []
      image_fcnt = []
      image_olen = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_test[key_str]['videopath'])
        image_fcnt.append(X_test[key_str]['framecnt'])
        image_olen.append(seq_len)
        is_training.append(False) # Testing
      image_info = zip(image_path,image_fcnt,image_olen,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_rgb_data)
      if modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_flow_data)
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)

