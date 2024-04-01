from model.setting import Setting, Arguments
from model.ourcse.processor import Processor
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]= "2,3" 

def main(args, logger) -> None:
    processor = Processor(args)
    config = processor.model_setting()
    logger.info('Model Setting Complete')

    if args.train == 'True':
        logger.info('Start Training')
        
        for epoch in range(args.epochs):
            print(f"epoch {epoch}")
            processor.train(epoch+1)

    if args.test == 'True':
        logger.info("Start Test")
        
        processor.test()
        processor.metric.print_size_of_model(config['model'])
        processor.metric.count_parameters(config['model'])


if __name__ == '__main__':
    args, logger = Setting().run()
    main(args, logger)