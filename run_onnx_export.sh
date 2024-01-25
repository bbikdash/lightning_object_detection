# Convert a standard YOLOX model by -f. When using -f, the above command is equivalent to:
python3 tools/export_onnx.py --output-name yolox_s.onnx -f exps/default/yolox_s.py -c yolox_s.pth

# To convert your customized model, use -f:
# python3 tools/export_onnx.py --output-name your_yolox.onnx -f exps/your_dir/your_yolox.py -c your_yolox.pth
python3 tools/export_onnx.py --output-name your_yolox.onnx -f exps/uav/your_yolox.py -c your_yolox.pth
