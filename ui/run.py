import argparse
import sys
sys.path.append('/home/yanxl/Aigc/ER-nerf')
from ui.ui_predict import ernerf_demo_infer
from ui.ui_train import ernerf_demo_train

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None) 
    parser.add_argument("--server", type=str, default='127.0.0.1')
    parser.add_argument("--is_train",action="store_true",help = "train ui") 
    args = parser.parse_args()
    demo = None
    if(args.is_train):
        demo = ernerf_demo_train()
    else:
        demo = ernerf_demo_infer()
    demo.queue()
    demo.launch(server_name=args.server, server_port=args.port)
