# GC-Net
Geometry and Context Network


### Requirements
- First, Use `generate_image_list.py` to generate an image list file, e.g. `train.lst`, which records the absolute image file paths.

- `readPFM.py` is utilized for read `PFM` files since the Disparity Groundtruth of Scene Flow - flyingthings3d Dataset is .pfm files.

### How to run
```
   CUDA_VISIBLE_DEVICES=n python main.py --gpu n --phase train --max_steps 50000 --learning_rate 0.00005 --output_dir /home/users/shixin.li/segment/gc-net/log/0508 --pretrain true
```
   `n` is the index of gpu. If there is no pretrained model, `pretrain` option should be `false`

### Something Else
It is said that the original version has been tested 20K steps by Scene Flow. Convergence effect is not good no matter whether the input files are RGB or Grayscale files.

The memory of 1080 GPU is not sufficient for the modified version. 

	ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[1,96,270,480,64]	 [[Node: transpose = Transpose[T=DT_FLOAT, Tperm=DT_INT32, _device="/job:localhost/replica:0/task:0/gpu:0"](Reshape, transpose/perm)]]	 [[Node: AddN/_85 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_561636_AddN", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

### Citation


	@article{Kendall2017EndtoEndLO,
	  title={End-to-End Learning of Geometry and Context for Deep Stereo Regression},
	  author={Alex Kendall and Hayk Martirosyan and Saumitro Dasgupta and Peter Henry and Ryan Kennedy and Abraham Bachrach and Adam Bry},
	  journal={CoRR},
	  year={2017},
	  volume={abs/1703.04309}
	}

