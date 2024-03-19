<<<<<<< HEAD
from model.setting import Setting, Arguments
from model.ourcse.processor import Processor
# ohjihyeon
=======
# from model.setting import Setting, Arguments
# from model.ourcse.processor import Processor

>>>>>>> 755f9f53a0cd64fc8388fdc2c493f01af2a53b3c

def main(args, logger) -> None:
    processor = Processor(args)
    config = processor.model_setting()
    logger.info('Model Setting Complete')

    if args.train == 'True':
        logger.info('Start Training')
        
        for epoch in range(args.epochs):
            processor.train(epoch+1)

    if args.test == 'True':
        logger.info("Start Test")
        
        processor.test()
        processor.metric.print_size_of_model(config['model'])
        processor.metric.count_parameters(config['model'])


if __name__ == '__main__':
    args, logger = Setting().run()
    main(args, logger)