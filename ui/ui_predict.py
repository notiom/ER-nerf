import os, sys
sys.path.append('/home/yanxl/Aigc/ER-nerf/')
import argparse
import gradio as gr
from nerf_triplane.provider import NeRFDataset
from nerf_triplane.utils import *
from nerf_triplane.network import NeRFNetwork
from scripts.adddudio import video_add_audio

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except AttributeError as e:
    print('Info. This pytorch version is not support with tf32.')

class MyPredict(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = [PSNRMeter(), LPIPSMeter(device=self.device), LMDMeter(backend='fan')]
        self.opt = inps = {
            'path': None,
            'O': False,  
            'test': False,
            'test_train': False,
            'data_range': [0, -1],
            'workspace': 'workspace',
            'seed': 0,
            'iters': 2000000,
            'lr': 1e-2,
            'lr_net': 1e-3,
            'ckpt': 'latest',
            'num_rays': 4096 * 16,
            'cuda_ray': False,
            'max_steps': 16,
            'num_steps': 16,
            'upsample_steps': 0,
            'update_extra_interval': 16,
            'max_ray_batch': 4096,
            'warmup_step': 10000,
            'amb_aud_loss': 1,
            'amb_eye_loss': 1,
            'unc_loss': 1,
            'lambda_amb': 1e-4,
            'fp16': False,
            'bg_img': '',
            'fbg': False,
            'exp_eye': False,
            'fix_eye': -1,
            'smooth_eye': False,
            'torso_shrink': 0.8,
            'color_space': 'srgb',
            'preload': 0,
            'bound': 1,
            'scale': 4,
            'offset': [0, 0, 0],
            'dt_gamma': 1/256,
            'min_near': 0.05,
            'density_thresh': 10,
            'density_thresh_torso': 0.01,
            'patch_size': 1,
            'init_lips': False,
            'finetune_lips': False,
            'smooth_lips': False,
            'torso': False,
            'head_ckpt': '',
            'gui': False,
            'W': 450,
            'H': 450,
            'radius': 3.35,
            'fovy': 21.24,
            'max_spp': 1,
            'att': 2,
            'aud': '',
            'emb': False,
            'ind_dim': 4,
            'ind_num': 10000,
            'ind_dim_torso': 8,
            'amb_dim': 2,
            'part': False,
            'part2': False,
            'train_camera': False,
            'smooth_path': False,
            'smooth_path_window': 7,
            'asr': False,
            'asr_wav': '',
            'asr_play': False,
            'asr_model': 'deepspeech',
            'asr_save_feats': False,
            'fps': 25,
            'l': 10,
            'm': 50,
            'r': 10
            }
        
        self.keys = [
                    'asr_wav',
                    'aud',
                    'asr_model',
                    'att',
                    'O',
                    'torso',
                    'test',
                    'test_train',
                    'finetune_lips',
                    'emb',
                    'smooth_path',
                    'smooth_eye',
                    'smooth_lips',
                    'train_camera',
                    'asr_play',
                    'asr_save_feats',
                    'smooth_path_window',
                    'path',
                    'workspace'
                    ]
        self.result_path = None  #工作空间的结果
        self.output_path = None  #最终结果
    def grinit(self, *args):
        info = ""
        print(f"args {args}")
        try: # 去运行
            for key_index in range(len(self.keys)):
                self.opt[self.keys[key_index]] = args[key_index]
                
            if(isinstance(self.opt['path'],str) is False):
                self.opt['path'] = self.opt['path'][0]
                    
            if(self.opt['path'] == None or os.path.exists(self.opt['path']) is False):
                #说明路径不存在，报错
                print(f"找不到输入数据集路径 {self.opt['path']}")
                raise FileNotFoundError
                
            if(isinstance(self.opt['workspace'],str) is False):
                self.opt['workspace'] = self.opt['workspace'][0]
                    
            if(self.opt['workspace'] == None or os.path.exists(self.opt['workspace']) is False):
                #说明路径不存在，报错
                print(f"找不到工作空间路径 {self.opt['workspace']}")
                raise FileNotFoundError
            
            if(self.opt['aud'] == None or os.path.exists(self.opt['aud']) is False):
                #说明路径不存在，报错
                print(f"找不到音频npy文件路径 {self.opt['aud']}")
                raise FileNotFoundError
            self.opt = argparse.Namespace(**self.opt)
            try:
                self.run()                   
            except Exception as e:
                # 运行推理错误
                content = f"{e}"
                info = f"Inference ERROR: {content}"
        except Exception as e:
            if info == "": 
                # 网页ui错误
                content = f"{e}"
                info = f"WebUI ERROR: {content}"
        
        # output part
        if len(info) > 0 :
            # errors    
            print(info)
            info_gr = gr.update(visible=True, value=info)
        else: 
            # no errors
            info_gr = gr.update(visible=False, value=info)
        output_path_folder = os.path.join(self.opt.get('workspace'),'results')
        for file in os.listdir(output_path_folder):
            if file[-5]!='h':
                self.result_path = os.path.join(output_path_folder, file)
                self.output_path = video_add_audio(self.result_path,self.opt.get('asr_wav'))
                     
        if self.output_path is not None and len(self.output_path) > 0 and os.path.exists(self.output_path): # good output
            print(f"成功生成 {self.output_path}!")
            video_gr = gr.update(visible=True, value=self.output_path)
        else:
            print(f"生成失败!")
            video_gr = gr.update(visible=True, value=self.output_path)          
        return video_gr, info_gr
    
    def run(self):
        if self.opt.O:
            self.opt.fp16 = True
            self.opt.exp_eye = True     
        self.opt.cuda_ray = True
        # assert opt.cuda_ray, "Only support CUDA ray mode."
        if self.opt.patch_size > 1:
            # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
            assert self.opt.num_rays % (self.opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."
        
        if self.opt.finetune_lips:
            # do not update density grid in finetune stage
            self.opt.update_extra_interval = 1e9
        
        print(f"opt {self.opt}")
        
        seed_everything(self.opt.seed)
        
        model = NeRFNetwork(self.opt)
        
        criterion = torch.nn.MSELoss(reduction='none')

        trainer = Trainer('ngp', self.opt, model, device=self.device, workspace=self.opt.workspace, criterion=criterion, fp16=self.opt.fp16, metrics=self.metrics, use_checkpoint=self.opt.ckpt)

        test_set = NeRFDataset(self.opt, device=self.device, type='train')
        # a manual fix to test on the training dataset
        test_set.training = False 
        test_set.num_rays = -1
        test_loader = test_set.dataloader()

        model.aud_features = test_loader._data.auds
        model.eye_areas = test_loader._data.eye_area

        ### test and save video (fast)  
        trainer.test(test_loader)

        ### evaluate metrics (slow)
        if test_loader.has_gt:
            trainer.evaluate(test_loader)
        # temp fix: for update_extra_states
        model.aud_features = test_loader._data.auds
        model.eye_areas = test_loader._data.eye_area
        ### test and save video (fast)  
        trainer.test(test_loader)
        ### evaluate metrics (slow)
        if test_loader.has_gt:
            trainer.evaluate(test_loader)
    
def ernerf_demo_infer():
    sep_line = "-" * 40
    infer_obj = MyPredict()
    print(sep_line)
    with gr.Blocks(analytics_enabled=False) as ernerf_interface:
        gr.Markdown("<div align='center'> <h2> ER-NERF: infer </span> </h2> </div>")     
        
        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="driven_audio"):
                    with gr.TabItem('Upload audio'):
                        with gr.Column(variant='panel'):
                            #从本上传要推理的音频
                            asr_wav = gr.FileExplorer(glob="/**/*.wav", file_count='single', root_dir = '/home/yanxl/Aigc/ER-nerf/data/test_audio/',label='load .wav file to predict',interactive=True)
                            aud = gr.FileExplorer(glob="/**/*.npy", file_count='single',root_dir = '/home/yanxl/Aigc/ER-nerf/data/test_audio/', label='load .npy file to predict',interactive=True)

                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem('path'):
                        with gr.Column(variant='panel'):
                            ckpt_info_box = gr.Textbox(value="Please select \"datasets\" and \"workspace\" under the checkpoint folder ", interactive=False, visible=True, show_label=False)
                                         
                            path = gr.FileExplorer(glob = "/**", ignore_glob= "/*.*",file_count='multiple', root_dir = '/home/yanxl/Aigc/ER-nerf/data/',label='data path directory',interactive=True)
                            workspace = gr.FileExplorer(glob = "/**", ignore_glob= "/*.*", file_count='multiple',root_dir = '/home/yanxl/Aigc/ER-nerf/Myworkspace/', label='torso model ckpt path or directory',interactive=True)
                                       
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem('Parameters Settings'):
                        with gr.Column(variant='panel'):
                            
                            asr_model = gr.Radio(['deepspeech', 'hubert'], value='deepspeech', label='asr_model', info="the method of audio extraction")
                            att = gr.Radio([0, 1, 2], value= 2, label='att', info="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
        
                            Oi = gr.Checkbox(label='O', value = True, info="equals --fp16 --cuda_ray --exp_eye",interactive=False) # 不允许该值被修改  
                            torso = gr.Checkbox(label='torso', value = True,info="fix head and train torso",interactive=False) # 不允许该值被修改    
                            test = gr.Checkbox(label='test', value = True,info="test mode (load model and test dataset)",interactive=False) # 不允许该值被修改 
                            test_train = gr.Checkbox(label='test_train', value = True,info="test mode (load model and train dataset)",interactive=False) # 不允许该值被修改 
                            
                            finetune_lips = gr.Checkbox(label='finetune_lips', value = False,info="use LPIPS and landmarks to fine tune lips region")
                            emb = gr.Checkbox(label='emb', value = False,info="use audio class + embedding instead of logits")   
                            #smooth                   
                            smooth_path = gr.Checkbox(label='smooth_path', value = False, info="brute-force smooth camera pose trajectory with a window size")
                            smooth_eye = gr.Checkbox(label='smooth_eye', value = False, info="smooth the eye area sequence")
                            smooth_lips = gr.Checkbox(label='smooth_lips', value = False, info="smooth the enc_a in a exponential decay way...")
                            train_camera = gr.Checkbox(label='train_camera', value = False, info="optimize camera pose")
                            # asr 设置
                            asr_play = gr.Checkbox(label='asr_play', value = False, info="play out the audio") 
                            asr_save_feats = gr.Checkbox(label='asr_save_feats', value = False) 
                            
                            smooth_path_window = gr.Slider(minimum=1, maximum=10, step=1, label="smooth_path_window", value=7, info='smoothing window size')
                            submit = gr.Button('Generate', elem_id="generate", variant='primary')
                        with gr.Column(variant='compact'):
                            with gr.Tabs(elem_id="genearted_video"):
                                    info_box = gr.Textbox(label="Error", interactive=False, visible=False)
                                    gen_video = gr.Video(label="Generated video", format="mp4", visible=True)
               
        fn = infer_obj.grinit
        submit.click(
                    fn=fn, 
                    inputs = [
                            asr_wav,
                            aud,
                            asr_model,
                            att,
                            Oi,
                            torso,
                            test,
                            test_train,
                            finetune_lips,
                            emb,
                            smooth_path,
                            smooth_eye,
                            smooth_lips,
                            train_camera,
                            asr_play,
                            asr_save_feats,
                            smooth_path_window,
                            path,
                            workspace
                            ],
                    outputs=[
                        gen_video,
                        info_box,
                    ],
                    )

    print(sep_line)
    print("Gradio page is constructed.")
    print(sep_line)

    return ernerf_interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None) 
    parser.add_argument("--server", type=str, default='127.0.0.1') 
    args = parser.parse_args()
    demo = ernerf_demo_infer()
    demo.queue()
    demo.launch(server_name=args.server, server_port=args.port)
