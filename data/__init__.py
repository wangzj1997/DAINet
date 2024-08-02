from importlib import import_module
from data.dataloader import MSDataLoader


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())     ## load the right dataset loader module                          data_train==RefMRI
            trainset = getattr(module_train, args.data_train)(args, name=args.name_train)             ## load the dataset, args.data_train is the  dataset name               name_train==mattest
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory= args.cpu
            )

        module_test = import_module('data.' + args.data_test.lower())
        testset = getattr(module_test, args.data_test)(args, name=args.name_test, train=False)

        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=args.cpu
        )

