from utils import *
import argparse
from configs import runCFG
from constants import *
from inference import show_final_result_, get_dataloader_

def parser():
    # Initialize the Parser
    parser = argparse.ArgumentParser()

    # Adding Arguments
    parser.add_argument('-d','--dataset', type=str, required=True,
                        help='Enter the dataset name')

    parser.add_argument('-m','--mode', type=str,
                        required=False, default='test', help='train, test, load')

    parser.add_argument('-t','--time', type=str,
                        required=False, default='none', help='specify time')
    
    parser.add_argument('-px', '--postfix', type=str, required=False,
                        default='none', help='the postfix of saved model name')

    parser.add_argument('-model','--model', type=str,
                        required=True, help='model name')

    parser.add_argument('-c','--cuda', type=str,
                        required=False, default=0, help='cuda device id')

    parser.add_argument('-ft','--feature_type', type=str,
                        required=False, default=feature_type_list[1], help=f'Select from {feature_type_list}')
    
    parser.add_argument('-mw', '--model_weights', type=str, required=False, default='none',
                        help=f'Location of the .pt file to load weights from. Don\'t use this argument if you are not finetuning')
    
    parser.add_argument('-k', '--topk', type=int, required=False, default=1,
                        help=f'topk value, defaults to k=1')

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parser()

    cfg = runCFG(args.dataset, args.model, args.postfix, args.feature_type, args.mode, args.time, args.model_weights, args.cuda)

    # input to the model
    print(cfg.dataset_name)

    print("Model Name:-", cfg.save_model_path)

    # model, loss function and optimizer
    model = load_model(cfg)
    print('total trainable params :',sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = get_criterion(cfg)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    
    early_stopper = EarlyStopper(cfg.early_stopping_step, 0) if cfg.early_stopping else None

    print("Running Mode:-", args.mode)      

    if args.mode in ["train", "load"]:
        # dataset & dataloader
        train_loader, val_loader, test_loader = get_dataloader(cfg)

        save_config_model_transform_params(cfg.save_config_path, cfg)

        # main_loop
        run_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg, early_stopper = early_stopper)

    cfg.mode = "test"
    # cfg.batch_size = 1

    # dataset & dataloader
    train_loader, val_loader, test_loader = get_dataloader(cfg)
    # train_loader, val_loader, test_loader = get_dataloader_(cfg)


    # load best model states and test
    # show_final_result_(model, train_loader, val_loader, test_loader, criterion, cfg, args.topk)
    show_final_result(model, train_loader, val_loader, test_loader, criterion, cfg, args.topk)
    gen_cm_w(model, test_loader, cfg)
