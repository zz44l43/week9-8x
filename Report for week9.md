# Report for week9

#### Code modification for - convert_fcn_dataset.py

This section of code is to link the classes and color map. So in the end, it will generate a TFRecord file for TesnsorFlow to consume and process.

```python
import sys
sys.path.insert(0,'../week8_homework/homework/object_detection')
from object_detection.utils import dataset_util
```

Re-use dataset_utils from previous week's code - so we can make our life easier. Thus we need sys pkg.

~~~~python
feature_dict = {
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(
        os.path.split(data)[1].encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_data),
    'image/label': dataset_util.bytes_feature(encoded_label),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
}
~~~~

In the dict_to_tf_example definition we full fill the feature dictionary.

Full fill the properties of the features

~~~~python
def create_tf_record(output_filename, file_pars):
    # Your code here
    writer = tf.python_io.TFRecordWriter(output_filename)
    for (data,label) in file_pars:
        print(data, label)
        tf_example = dict_to_tf_example(data, label)
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())
    writer.close()
~~~~

Make sure the writer output the files to the directory



#### Code modification for - convert_fcn_dataset.py

The purpose of this section of code is to convert fcn16x to fcn8x.

~~~~python
upsample_factor = 8
~~~~

Change the upsample factor from 16 to 8. So that network only requires to backwards convolution 8 times in the end rather than 16 times. It is simply reverses the forwards and backward passes of convolution.

~~~~python
pool3_feature = end_points['vgg_16/pool3']
~~~~

Add the above line of code.

1 of main differences between 16x and 8x is that 16x start the skips between the layers to fuse coarse from pool4. However, the 8x starts from the pool3 of the vgg.



Keep all the fcn16x code except the following one:

```python
upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x16,
                                          output_shape=upsampled_logits_shape,
                                          strides=[1, upsample_factor, upsample_factor, 1],
                                          padding='SAME')
```
Because in fcn8x we need upsample by 8 rather than 16 in the end. So we deleted the above.



The additional code for the fcn8x started from the following:

~~~~python
#Add 1x1 conv to the pool3 of Vgg16 and call it score_pool3. Thus produce an additional class predictions. There are new parameters added here but initialed with zeros thus the net starts with unmodified prediction.
with tf.variable_scope('vgg_16/fc8'):
    logging.debug('score pool3...')
    aux_logits_8s = slim.conv2d(pool3_feature, number_of_classes, [1, 1],
                             activation_fn=None,
                             weights_initializer=tf.zeros_initializer,
                             scope='conv_pool3') #score_pool3

upsample_filter_np_xx2 = bilinear_upsample_weights(2,  # upsample_factor,
                                                  number_of_classes)
upsample_filter_tensor_xx2 = tf.Variable(upsample_filter_np_xx2, name='vgg_16/fc8/t_conv_xx2')

#We fuse this score_pool3 at stride of [1,2,2,1] and adding a 2x upsampling to bilinear interpolation but allow the parameter to be learned. Thus produced score_pool3c
logging.debug('score pool3c...')
upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_xx2,
                                          output_shape=tf.shape(aux_logits_8s),
                                          strides=[1, 2, 2, 1],
                                          padding='SAME') #score_pool3c

#Furthermore, we add up with the predictions (score_ pool3) to the score_pool3c.
logging.debug('fuse pool3...')
upsampled_logits = upsampled_logits + aux_logits_8s #fuse_pool3

upsample_filter_np_x8 = bilinear_upsample_weights(upsample_factor,
                                                   number_of_classes)
upsample_filter_tensor_x8 = tf.Variable(upsample_filter_np_x8, name='vgg_16/fc8/t_conv_x8')

#Finally, the stride 8 predictions are upsampled back to the image.
logging.debug('fuse 8x...')
upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x8,
                                          output_shape=upsampled_logits_shape,
                                          strides=[1, upsample_factor, upsample_factor, 1],
                                          padding='SAME')
~~~~

Add 1x1 conv to the pool3 of Vgg16 and call it score_pool3. Thus produce an additional class predictions. There are new parameters added here but initialed with zeros thus the net starts with unmodified prediction.

We fuse this score_pool3 at stride of [1,2,2,1] and adding a 2x upsampling to bilinear interpolation but allow the parameter to be learned. Thus produced score_pool3c

Furthermore, we add up with the predictions (score_ pool3) to the score_pool3c.

Finally, the stride 8 predictions are upsampled back to the image.



#### Results:

##### Dataset:

Url: https://www.tinymind.com/danni0813/datasets/week9

##### Content:

* fcn_val.record - 399.98MB
* fcn_train.record - 403.41MB
* vgg_16.ckpt - 527.8MB

Executing the run produces the following - Execution 1:

**log:**

~~~~python
2018-02-18 15:59:06,203 - DEBUG - train.py:322 - step 1380 Current Loss: 112.78339385986328 
2018-02-18 15:59:06,203 - DEBUG - train.py:324 - [9.92] imgs/s
2018-02-18 15:59:22,601 - DEBUG - train.py:322 - step 1390 Current Loss: 99.02734375 
2018-02-18 15:59:22,601 - DEBUG - train.py:324 - [9.76] imgs/s
2018-02-18 15:59:39,192 - DEBUG - train.py:322 - step 1400 Current Loss: 89.30299377441406 
2018-02-18 15:59:39,192 - DEBUG - train.py:324 - [9.64] imgs/s
2018-02-18 15:59:41,042 - DEBUG - train.py:331 - Model saved in file: /output/train/model.ckpt-1400
2018-02-18 15:59:41,043 - DEBUG - train.py:338 - validation generated at step [1400]
2018-02-18 15:59:58,950 - DEBUG - train.py:322 - step 1410 Current Loss: 75.853271484375 
2018-02-18 15:59:58,950 - DEBUG - train.py:324 - [8.10] imgs/s
2018-02-18 16:00:15,966 - DEBUG - train.py:322 - step 1420 Current Loss: 98.62287139892578 
2018-02-18 16:00:15,966 - DEBUG - train.py:324 - [9.40] imgs/s
2018-02-18 16:00:32,539 - DEBUG - train.py:322 - step 1430 Current Loss: 88.7820053100586 
2018-02-18 16:00:32,539 - DEBUG - train.py:324 - [9.65] imgs/s
2018-02-18 16:00:48,897 - DEBUG - train.py:322 - step 1440 Current Loss: 113.26451110839844 
2018-02-18 16:00:48,897 - DEBUG - train.py:324 - [9.78] imgs/s
2018-02-18 16:01:05,085 - DEBUG - train.py:322 - step 1450 Current Loss: 140.9359893798828 
2018-02-18 16:01:05,085 - DEBUG - train.py:324 - [9.88] imgs/s
2018-02-18 16:01:21,508 - DEBUG - train.py:322 - step 1460 Current Loss: 90.7876968383789 
2018-02-18 16:01:21,508 - DEBUG - train.py:324 - [9.74] imgs/s
2018-02-18 16:01:38,097 - DEBUG - train.py:322 - step 1470 Current Loss: 68.58013153076172 
2018-02-18 16:01:38,098 - DEBUG - train.py:324 - [9.64] imgs/s
2018-02-18 16:01:54,426 - DEBUG - train.py:322 - step 1480 Current Loss: 89.13813018798828 
2018-02-18 16:01:54,426 - DEBUG - train.py:324 - [9.80] imgs/s
2018-02-18 16:02:10,907 - DEBUG - train.py:322 - step 1490 Current Loss: 69.51349639892578 
2018-02-18 16:02:10,908 - DEBUG - train.py:324 - [9.71] imgs/s
2018-02-18 16:02:27,336 - DEBUG - train.py:322 - step 1500 Current Loss: 64.83626556396484 
2018-02-18 16:02:27,337 - DEBUG - train.py:324 - [9.74] imgs/s
2018-02-18 16:02:29,171 - DEBUG - train.py:331 - Model saved in file: /output/train/model.ckpt-1500
2018-02-18 16:02:45,180 - DEBUG - train.py:357 - Model saved in file: /output/train/model.ckpt-1500
~~~~

A section of log is shown above to demonstrate that the running was successful and appropriate model ckpt is saved.

**The parameter setting is the following for the above log:**

~~~~
batch_size: 16
output_dir: /output
checkpoint_path: /data/danni0813/week9/vgg_16.ckpt
dataset_train: /data/danni0813/week9/fcn_train.record
dataset_val: /data/danni0813/week9/fcn_val.record
~~~~

![eval/val_1000_img.jpg](https://storage.googleapis.com/tinymind/execs%2F5q0vgwzu%2Foutput%2Feval%2Fval_1000_img.jpg?GoogleAccessId=production%40colafly-tinymind.iam.gserviceaccount.com&Expires=1519004501&Signature=v1b7I7zMH1%2ByAJRB87Igf1RTzIGVrKAXTbtFabe5i7INxLoW%2B9T47KRSjajGCvtazm4pl5QnwvSJC7FMj8Lop91ExC%2FumUDPWYBWKZDzgczJTIC887SdLSSR46Fbr1ZCi1Jr2D4JpO%2F52%2BQkxqd9aIkYnUuQESGTzAJuEQVh4jUuo5mKdFJsMPNZsNxreMA5lEpZ%2Fml%2BRNXSG9vQnIcsx%2B8469BZAXhvTKjDAjC%2BMFeSEQ%2Fz%2FHeAV44bBvuGR7CroLnKDCcmjTCaCsQ7F8FNt9X7%2BncEgS3wE58uJs8cNvBg2isfvV5obs662mJemvLuouqR%2B69ccsEc9MUyQc5dbw%3D%3D)



![eval/val_1000_annotation.jpg](https://storage.googleapis.com/tinymind/execs%2F5q0vgwzu%2Foutput%2Feval%2Fval_1000_annotation.jpg?GoogleAccessId=production%40colafly-tinymind.iam.gserviceaccount.com&Expires=1519004482&Signature=VbMQkF30S2R1HzZnJCeSUVA%2Bc1iYJO7SY95p5E7KNqBQEpy6wKCvcp7%2FQvla38Ew5zdsYflPNs%2B%2FvXkdnTuoCbQZvQwWFLVp872PsNnVtLVXvWiruaQ%2BRmJ1qLw1%2FjYALs5KFp7Ab96lfFy4E6rQG1MmW84h4LlhmoSdGjxdVx%2Fak98lllXcvH60dbBzMOk%2BJ74aFvjThhLxcOhywnl5YuqQDYYqWcskbm%2F62mqjnuvaUGvzCcsSbLANTIQIGNcVEm6bHKO5Yx%2Bzg8sj%2FXGu1%2FEYYME2MfQ5CJx9nntokd5tc8ntphxrrQMNn7KQ6a%2FKQzg%2Bcn%2F%2BprO04xcRUl8h9w%3D%3D)

![eval/val_1000_prediction.jpg](https://storage.googleapis.com/tinymind/execs%2F5q0vgwzu%2Foutput%2Feval%2Fval_1000_prediction.jpg?GoogleAccessId=production%40colafly-tinymind.iam.gserviceaccount.com&Expires=1519004528&Signature=V%2FcK5w4Z%2B8wWxsCCO%2FRTZ%2FThdWYAzFwup8SnvnlSOMuwGFtxh1Y8yEhYsIjC64JRxnp5lrkmrjR0yrHELXi3a7R5lyz6V8KD2EW15pyJDHMiMAi3Z1oTHcv7FKc1lqYKCLP6a%2BEL5lEaOUWiJHSK6TmngMOXnpNrjXTmX7LPv9J0QSwSMpGVP7E34sQ2CmL%2B2hSYMVz3J3p5k6N4e7DRrHeDWMr52cFKX1X5HqUuCOlEQrxmhTSDNRoNGa5KxaetkjPRLYVabYv%2BKLzUh4j4j9VKWM8wS5Ydhk0U8PMNXhtl1y00yeicrqqwU6djxXXNvHnwiIBIiiw9wqE%2BniJ94Q%3D%3D)

![eval/val_1000_prediction_crfed.jpg](https://storage.googleapis.com/tinymind/execs%2F5q0vgwzu%2Foutput%2Feval%2Fval_1000_prediction_crfed.jpg?GoogleAccessId=production%40colafly-tinymind.iam.gserviceaccount.com&Expires=1519004546&Signature=A0P9zE%2Bji8E5ww3ceKArBLHkE%2BDN0hmNfpjIPazJwwyxY64oKPniW29XIO13jHD%2F0j%2B%2FqOKI%2BokvaX4qRLivCo2xsGCI1ARXDuM%2Bj4%2F4hKursjkPvNTpbTIkbSLQsfDycRifsPx9vo8q1Vi3eEM%2BJrc%2BCYai3ycNUvBGpDdqoJOHFfyl62wN4JTpEiVFV4Mi%2FNVJLuUnCUJ5%2BGrbXQJzXPaQCfHCSOXeDcxBkGCZSZGEbi0bA4sXA6eCosanC95uGiZtp0oA8goeZDpLyXbOn%2BJRad2qLqq1AyC3h0peKzkUhMGe9G112YNBpoSvfLf5BYVioNOTnZ57u4M8J7LFOQ%3D%3D)

![eval/val_1000_overlay.jpg](https://storage.googleapis.com/tinymind/execs%2F5q0vgwzu%2Foutput%2Feval%2Fval_1000_overlay.jpg?GoogleAccessId=production%40colafly-tinymind.iam.gserviceaccount.com&Expires=1519004587&Signature=VQbWlG8fkxf8lfRa%2F6Fe24gpM3N8kpQ%2BDmGBI4E%2F1u5jLaH%2Bo2ath2lRSNBW2KofgivTzxcSqTdq%2BSBfhINHk4JvxMYEJam1jXIezAtLv%2FbsY84PwS1gzUm5u7E5NX1tq4yxAfCVj6sSGKJdPprT1WLHJYGiNadidMTVIrussm2jlSmP%2Bxe7D12DkxprgbVrevBpX9XBYVJ%2BoCxoB3%2FiwEZ%2BT7UJrA5PuOMmgl090GGW%2Bt2qs3yh774E5H2sPdrYXNXf798oxgM3okpUiTxeBUnVh2fvK38CUeA7LJvG%2F3FBQSt2Y85jQd8jOHiY2L6xqQCjU%2Basb3XNEYjT4lUMZw%3D%3D)

The above is the evaluation result for this execution.



Execution2 

~~~~python
2018-02-19 02:46:53,525 - DEBUG - train.py:322 - step 1390 Current Loss: 58.772987365722656 
2018-02-19 02:46:53,525 - DEBUG - train.py:324 - [11.45] imgs/s
2018-02-19 02:47:21,337 - DEBUG - train.py:322 - step 1400 Current Loss: 77.37055969238281 
2018-02-19 02:47:21,337 - DEBUG - train.py:324 - [11.51] imgs/s
2018-02-19 02:47:23,254 - DEBUG - train.py:331 - Model saved in file: /output/train/model.ckpt-1400
2018-02-19 02:47:23,254 - DEBUG - train.py:338 - validation generated at step [1400]
2018-02-19 02:47:52,427 - DEBUG - train.py:322 - step 1410 Current Loss: 47.03799819946289 
2018-02-19 02:47:52,427 - DEBUG - train.py:324 - [10.29] imgs/s
2018-02-19 02:48:20,060 - DEBUG - train.py:322 - step 1420 Current Loss: 65.40450286865234 
2018-02-19 02:48:20,061 - DEBUG - train.py:324 - [11.58] imgs/s
2018-02-19 02:48:47,591 - DEBUG - train.py:322 - step 1430 Current Loss: 51.28940963745117 
2018-02-19 02:48:47,591 - DEBUG - train.py:324 - [11.62] imgs/s
2018-02-19 02:49:15,245 - DEBUG - train.py:322 - step 1440 Current Loss: 41.37165451049805 
2018-02-19 02:49:15,245 - DEBUG - train.py:324 - [11.57] imgs/s
2018-02-19 02:49:42,886 - DEBUG - train.py:322 - step 1450 Current Loss: 44.56869888305664 
2018-02-19 02:49:42,886 - DEBUG - train.py:324 - [11.58] imgs/s
2018-02-19 02:50:10,580 - DEBUG - train.py:322 - step 1460 Current Loss: 38.67291259765625 
2018-02-19 02:50:10,580 - DEBUG - train.py:324 - [11.55] imgs/s
2018-02-19 02:50:38,159 - DEBUG - train.py:322 - step 1470 Current Loss: 67.04952239990234 
2018-02-19 02:50:38,159 - DEBUG - train.py:324 - [11.60] imgs/s
2018-02-19 02:51:05,630 - DEBUG - train.py:322 - step 1480 Current Loss: 46.47848892211914 
2018-02-19 02:51:05,630 - DEBUG - train.py:324 - [11.65] imgs/s
2018-02-19 02:51:33,226 - DEBUG - train.py:322 - step 1490 Current Loss: 45.22264862060547 
2018-02-19 02:51:33,226 - DEBUG - train.py:324 - [11.60] imgs/s
2018-02-19 02:52:00,915 - DEBUG - train.py:322 - step 1500 Current Loss: 71.33306121826172 
2018-02-19 02:52:00,915 - DEBUG - train.py:324 - [11.56] imgs/s
2018-02-19 02:52:02,817 - DEBUG - train.py:331 - Model saved in file: /output/train/model.ckpt-1500
2018-02-19 02:52:20,000 - DEBUG - train.py:357 - Model saved in file: /output/train/model.ckpt-1500
~~~~



Parameters are the following:

~~~~python
batch_size 32
output_dir /output
checkpoint_path /data/danni0813/week9/vgg_16.ckpt
dataset_train /data/danni0813/week9/fcn_train.record
dataset_val /data/danni0813/week9/fcn_val.record
~~~~

![eval/val_1000_img.jpg](https://storage.googleapis.com/tinymind/execs%2Ffut8p46y%2Foutput%2Feval%2Fval_1000_img.jpg?GoogleAccessId=production%40colafly-tinymind.iam.gserviceaccount.com&Expires=1519012147&Signature=0xqWNS%2BsYH5kStK0uOBYJ3wwrdJIH8761oKRUhd8CLKFPzArvtIwCRDpcTCcBXDvtDhC2sUbisEKj%2BTqLOub4082e0kXadJzzykIqKHEATS5cKamB6ueQLsx5dHNgGveRRVqi4XaESjDNwdWQegAUw64Xkv9KpPDPIfEQS91BaufTtuCPHs2iDz35LJYFmlS1DpE7kKaxGhg8egHRxUWHTQS%2FjtAjSed8WbrgMeYbDO57NIHPAwV5Q%2FZZTnqcRO5OjC9ihyFUQ5I8n9S1YZGUdM8nXdzePRVU7r1sbzfoTwJEQQztz8FCFRA6SmF1mFVOyHyVE3xqAu7V4qexf6o8w%3D%3D)



![eval/val_1000_annotation.jpg](https://storage.googleapis.com/tinymind/execs%2Ffut8p46y%2Foutput%2Feval%2Fval_1000_annotation.jpg?GoogleAccessId=production%40colafly-tinymind.iam.gserviceaccount.com&Expires=1519012162&Signature=IBgzO03ywscWXkwEcgFcW6on4wN2j4t0wODjSgnKwNdhwqkmqepSHDKZWsqLC6CW7i8mTxx%2FWq8uQ98AsRXv5ADncj%2BeBvraWz41aBirnhhlsw0A0haMm1m0TvRy3ondte4xuCPZ87rgZyuYs%2FgT0wpBYh3J2Vz2bsZ3km%2FsI1dF5Wf8hnTC6%2BIXcOGk9nnKki5kV26wnm3y8szNtgyp0z8HgeJJoLZ1pgh1lWatfWRz%2B1PKnCxZqXkKi66HCzfn%2FVdA%2FerRBaWxOrX7DSGHsd1RpZu2aD90GOy7kEn8HqG1GZhqV6QN3ccfseQGen8v7%2FlbSzeFPE03HirpLDbeHg%3D%3D)

![eval/val_1000_prediction.jpg](https://storage.googleapis.com/tinymind/execs%2Ffut8p46y%2Foutput%2Feval%2Fval_1000_prediction.jpg?GoogleAccessId=production%40colafly-tinymind.iam.gserviceaccount.com&Expires=1519012171&Signature=UuynorXiHRFfe4o0GVtYHnIXyd7lrqCWJCDlTbqW%2FddeB5ix%2F83QosneElRco3uLZdlufGgZGq1jvux0FGf0pJBqP4%2BeZ3ld5wIac1Eu6ot3pHWCeZvZR4m49q1jisVKXpBS4RmrM5LM1STfCqaeqv2GNWuIwQ4QPfTHX2piYAqk5G1B2fg7mNxKNd%2F9iB4GxCZ8BxF5sA83o1GQJBoKT9YmdR4empG7sDA1FwwlYQkumXQxXI0mEQGUM9Z7UtKTFPlZ6ZA%2FsD0iVXbw5a0Mb32NUHrrjw%2BcTc9QafLCEi%2BsxDKGYEDm18ilURI4ydti2EXF8BuRvNIWg4TrWRJNeg%3D%3D)

![eval/val_1000_prediction_crfed.jpg](https://storage.googleapis.com/tinymind/execs%2Ffut8p46y%2Foutput%2Feval%2Fval_1000_prediction_crfed.jpg?GoogleAccessId=production%40colafly-tinymind.iam.gserviceaccount.com&Expires=1519012179&Signature=AqS%2FDuEBZ%2BLFNgHUnyTNDv7ipytOedTG0PXOWViBv8Y913xiOQZCNB1elM6aw8rn35VoOezeQ8qMSmkRSeBweH25ad5M%2BvSVmKLIxesd9oE%2BRjqtq1viUP2XEbRjg5uUUkujTMeR%2BYw%2B9ceA%2FboYdP8bLkcE6yFSMvdNvhFb8x3j87NG0gEHwx62IeMyprgNwUofdecyxv4Q0csYThtcyBZkDSDCBpb4x5yhBniJigAB8zVT1G5xg3Ydmjuh4HAjXNQK6ZOA%2FjcOmdr8tX%2BGByEnNZ4EMTcpeGslUj7PVmbypMOEL%2BqlkYu%2FyVO9bghfNXtnmpAXyicYCfsX5%2F1KiQ%3D%3D)

![eval/val_1000_overlay.jpg](https://storage.googleapis.com/tinymind/execs%2Ffut8p46y%2Foutput%2Feval%2Fval_1000_overlay.jpg?GoogleAccessId=production%40colafly-tinymind.iam.gserviceaccount.com&Expires=1519012190&Signature=IGc7s2BAxO5hKYGemkUjZ7DOXy09%2Bg7TFd7ZGI%2FFCA6V6wpLRqsicSSiha%2B1nSglwAz3m%2BqiS%2Bi9Dtfbd%2BFfoWqUhBHFa%2FaGaqdg%2BPTWo3DA8DVVdfSnx2lasvVXpMKacHgxDdH2JB%2FMCd%2FGqUaBPT%2B55PTHTad3cn6WU%2Bfw5A6wUzAD6gFMRPcvmfApmELWV%2BRfmPe7UyYGBaoSOGeexLlro6X55G1PqgIj17yt22cGzYbGtVPHQfvz2A4uNjwbI9QWfa%2F1SdMp1eaD3NJDanGi8ehkzjINiin6OEJQAZzAKjGC5akXDMLW%2BEx%2F0MKxuK9yRZaqYMc2xYI%2Byo9xWQ%3D%3D)

Same iteration but larger batch size. The model had shown some more segment on the left front light of the car. Furthermore, it also shows the better prediction on the right side of the windows.