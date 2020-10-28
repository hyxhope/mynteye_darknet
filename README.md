# mynteye_darknet

读取小觅摄像头并用darknet进行实时目标检测

[darknet](https://github.com/AlexeyAB/darknet)

需要安装小觅的SDK

## Dependencies

在Ubuntu18.04下测试通过，支持x86，arm
- Cmake >= 3.12 (编译darknet需要)
- [Mynteye-SDK](https://github.com/slightech/MYNT-EYE-D-SDK)

### Build
```Bash
git clone https://github.com/hyxhope/mynteye_darknet.git
cd mynteye_darknet/
cmake .
make
./test
```

如果报错找不到-ldarknet，即是找不到动态链接库libdarknet.so，视情况重新编译darknet，或者修改CMakeList.txt
