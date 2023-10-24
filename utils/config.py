import yaml


class ConfigureCVS:
    def __init__(self, path: str) -> None:
        super(ConfigureCVS, self).__init__()
        file = open(path, 'r')
        fileData = file.read()
        file.close()
        ydata = yaml.load(fileData, Loader=yaml.FullLoader)
        self.SEED = ydata['SEED']
        self.angRes = ydata['angRes']
        self.channels = ydata['channels']
        self.nGroups = ydata['nGroups']
        self.nBlocks = ydata['nBlocks']
        self.trainsetDir = ydata['trainsetDir']
        self.testsetDir = ydata['testsetDir']
        self.modelName = ydata['modelName']
        self.dataList = ydata['dataList']
        self.inList = ydata['inList']
        self.batchSize = ydata['batchSize']
        self.lr = ydata['lr']
        self.nEpochs = ydata['nEpochs']
        self.nSteps = ydata['nSteps']
        self.gamma = ydata['gamma']
        self.resumeEpoch = ydata['resumeEpoch']
        self.modelSaveRoot = ydata['modelSaveRoot']
        self.logSaveRootDir = ydata['logSaveRootDir']
        self.resultsSaveRoot = ydata['resultsSaveRoot']


class ConfigureHLFSSR:
    def __init__(self, path: str) -> None:
        super(ConfigureHLFSSR, self).__init__()
        file = open(path, 'r')
        fileData = file.read()
        file.close()
        ydata = yaml.load(fileData, Loader=yaml.FullLoader)
        self.SEED = ydata['SEED']
        self.angRes = ydata['angRes']
        self.upFactor = ydata['upFactor']
        self.modelName = ydata['modelName']
        self.dataList = ydata['dataList']
        self.testList = ydata['testList']
        self.inList = ydata['inList']
        self.trainsetDir = ydata['trainsetDir']
        self.testsetDir = ydata['testsetDir']
        self.batchSize = ydata['batchSize']
        self.lr = ydata['lr']
        self.nEpochs = ydata['nEpochs']
        self.nSteps = ydata['nSteps']
        self.gamma = ydata['gamma']
        self.resumeEpoch = ydata['resumeEpoch']
        self.epiLambda = ydata['epiLambda']
        self.modelSaveRoot = ydata['modelSaveRoot']
        self.resultsSaveRoot = ydata['resultsSaveRoot']
        self.logSaveRootDir = ydata['logSaveRootDir']
        self.CVSModelPath = ydata['CVSModelPath']
        self.BDModelPath = ydata['BDModelPath']


class ConfigureBD:
    def __init__(self, path: str) -> None:
        super(ConfigureBD, self).__init__()
        file = open(path, 'r')
        fileData = file.read()
        file.close()
        ydata = yaml.load(fileData, Loader=yaml.FullLoader)
        self.SEED = ydata['SEED']
        self.angRes = ydata['angRes']
        self.upFactor = ydata['upFactor']
        self.modelName = ydata['modelName']
        self.dataList = ydata['dataList']
        self.trainsetDir = ydata['trainsetDir']
        self.testsetDir = ydata['testsetDir']
        self.batchSize = ydata['batchSize']
        self.lr = ydata['lr']
        self.nEpochs = ydata['nEpochs']
        self.nSteps = ydata['nSteps']
        self.gamma = ydata['gamma']
        self.resumeEpoch = ydata['resumeEpoch']
        self.modelSaveRoot = ydata['modelSaveRoot']
        self.logSaveRootDir = ydata['logSaveRootDir']
