# cudnn library

Extract the contents of cudnn___.tgz:

`tar -xvf cudnn___.tgz`

copy all the contents of the include directory in path/to/cuda/include, for example:

`sudo cp -r lib64/* /usr/local/cuda/lib`

copy all the contents of the lib64 directory in path/to/cuda/lib64, for example:

`sudo cp -r lib64/* /usr/local/cuda/lib64/`
