# GC-Net
Geometry and Context Network


### Requirements
- TensorFlow 1.1.0
- Python 2.7.6 (default, Oct 26 2016, 20:30:19) 
- [GCC 4.8.4] on Ubuntu 14.04
- Numpy 1.11.0
- PIL 4.1.1

### Related Work
- [LinHungShi / GCNetwork](https://github.com/LinHungShi/GCNetwork)

- [MaidouPP/gc_net_stereo](https://github.com/MaidouPP/gc_net_stereo)

### How to run
- First, Use `generate_image_list.py` to generate an image list file, e.g. `train.lst`, which records the absolute image file paths.

- `readPFM.py` is utilized for read `PFM` files since the Disparity Groundtruth of Scene Flow - flyingthings3d Dataset is .pfm files.

- `n` is the index of gpu. If there is no pretrained model, `pretrain` option should be `false`
```
CUDA_VISIBLE_DEVICES=n python main.py --gpu n --phase train --max_steps 50000 --learning_rate 0.00005 --output_dir $HOME/Applications/GC-Net/log/ --pretrain true
```

### Something Else
It is said that the original version has been tested 20K steps by Scene Flow. Convergence effect is not good no matter whether the input files are RGB or Grayscale files.

The memory of 1080 GPU is not sufficient for the modified version. 
```
Traceback (most recent call last):
  File "main.py", line 242, in <module>
    tf.app.run()
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/platform/app.py", line 48, in run
    _sys.exit(main(_sys.argv[:1] + flags_passthrough))
  File "main.py", line 237, in main
    train(dataset, args)
  File "main.py", line 154, in train
    output = sess.run(i, feed_dict={left_img: imgL, right_img: imgR, labels: imgGround})
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 778, in run
    run_metadata_ptr)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 982, in _run
    feed_dict_string, options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1032, in _do_run
    target_list, options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1052, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[1,96,270,480,64]
	 [[Node: transpose = Transpose[T=DT_FLOAT, Tperm=DT_INT32, _device="/job:localhost/replica:0/task:0/gpu:0"](Reshape, transpose/perm)]]
	 [[Node: AddN/_85 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_560929_AddN", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

Caused by op u'transpose', defined at:
  File "main.py", line 242, in <module>
    tf.app.run()
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/platform/app.py", line 48, in run
    _sys.exit(main(_sys.argv[:1] + flags_passthrough))
  File "main.py", line 237, in main
    train(dataset, args)
  File "main.py", line 72, in train
    logits = inference(left_img, right_img, is_training=True)
  File "/SSD2/jack/Applications/gc_net_stereo/network.py", line 200, in inference
    cost_vol = tf.transpose(cost_vol, [0, 1, 3, 4, 2])
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/array_ops.py", line 1270, in transpose
    ret = gen_array_ops.transpose(a, perm, name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_array_ops.py", line 3721, in transpose
    result = _op_def_lib.apply_op("Transpose", x=x, perm=perm, name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/op_def_library.py", line 768, in apply_op
    op_def=op_def)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 2336, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1228, in __init__
    self._traceback = _extract_stack()

ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[1,96,270,480,64]
	 [[Node: transpose = Transpose[T=DT_FLOAT, Tperm=DT_INT32, _device="/job:localhost/replica:0/task:0/gpu:0"](Reshape, transpose/perm)]]
	 [[Node: AddN/_85 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_560929_AddN", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]
```

### Citation


	@article{Kendall2017EndtoEndLO,
	  title={End-to-End Learning of Geometry and Context for Deep Stereo Regression},
	  author={Alex Kendall and Hayk Martirosyan and Saumitro Dasgupta and Peter Henry and Ryan Kennedy and Abraham Bachrach and Adam Bry},
	  journal={CoRR},
	  year={2017},
	  volume={abs/1703.04309}
	}

