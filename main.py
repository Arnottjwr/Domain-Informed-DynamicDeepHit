# import class to train model and run
# just use config.json to configure settings

from DynamicDeepHit.D3CKD import DomainInformedModel


if __name__ == '__main__':
    model = DomainInformedModel()
    model.main()