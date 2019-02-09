import torch
import copy


class AdversarialTrainerFactory:
    factories = {}

    def addFactory(id, modelFactory):
        AdversarialTrainerFactory.factories.put[id] = modelFactory
    addFactory = staticmethod(addFactory)
    # A Template Method:

    def createModel(id, netD=None, criterion=None):
        if id not in AdversarialTrainerFactory.factories:
            AdversarialTrainerFactory.factories[id] = \
              eval(id + '.Factory()')
        return AdversarialTrainerFactory.factories[id].create(netD, criterion)
    createModel = staticmethod(createModel)


class AdversarialTrainer(object):
    def __init__(self, netD, criterion):
        self.netD = netD
        self.criterion = criterion

    def lossD(self, pred, gt):
        pass

    def lossG(self, pred, gt):
        pass

    def get_params(self):
        pass


class NoAdversarialTrainer(AdversarialTrainer):
    def __init__(self, netD, criterion):
        AdversarialTrainer.__init__(self, netD, criterion)

    def lossD(self, pred, gt):
        return [0]

    def lossG(self, pred, gt):
        return 0

    def get_params(self):
        return [torch.nn.Parameter(torch.Tensor(1))]

    class Factory:
        def create(self, netD, criterion): return NoAdversarialTrainer(netD, criterion)


class SingleAdversarialTrainer(AdversarialTrainer):
    def __init__(self, netD, criterion):
        AdversarialTrainer.__init__(self, netD, criterion)
        self.netD = self.netD.cuda()

    def lossD(self, pred, gt):
        return self.criterion(self.netD, pred, gt)

    def lossG(self, pred, gt):
        return self.criterion.get_g_loss(self.netD, pred, gt)

    def get_params(self):
        return self.netD.parameters()

    class Factory:
        def create(self, netD, criterion): return SingleAdversarialTrainer(netD, criterion)


class DoubleAdversarialTrainer(AdversarialTrainer):
    def __init__(self, netD, criterion):
        AdversarialTrainer.__init__(self, netD, criterion)
        self.patchD = netD['patch']
        self.fullD = netD['full']
        self.patchD = self.patchD.cuda()
        self.fullD = self.fullD.cuda()
        self.full_criterion = copy.deepcopy(criterion)

    def lossD(self, pred, gt):
        return (self.criterion(self.patchD, pred, gt) + self.full_criterion(self.fullD, pred, gt)) / 2

    def lossG(self, pred, gt):
        return (self.criterion.get_g_loss(self.patchD, pred, gt) + self.full_criterion.get_g_loss(self.fullD, pred, gt)) / 2

    def get_params(self):
        return list(self.patchD.parameters()) + list(self.fullD.parameters())

    class Factory:
        def create(self, netD, criterion): return DoubleAdversarialTrainer(netD, criterion)

