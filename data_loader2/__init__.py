import torchvision.transforms as transforms

def dataset(dataset='mnist', mode='train', classes=[0,1,2,3,4,5,6,7,8,9]):
    if dataset == 'mnist':
        from .mnist import MNIST
        transform = transforms.Compose([
            # transforms.Scale(64),
            # transforms.Resize((32,32)),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5))
        ])
        if mode == 'train':
            return MNIST(root='/home/siit/data/mnist', train=True, download=True,
                         transform=transform, classes=classes)
        elif mode == 'test':
            return MNIST(root='/home/siit/data/mnist', train=False, download=True,
                         transform=transform, classes=classes)


    elif dataset == 'cifar10':
        from .cifar import CIFAR10
        if mode == 'train':
            transform = transforms.Compose([
                # transforms.Scale(64),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            return CIFAR10(root='/home/siit/data/cifar', train=True, download=True,
                           transform=transform, classes=classes)
        if mode == 'test':
            transform = transforms.Compose([
                # transforms.Scale(64),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
            return CIFAR10(root='/home/siit/data/cifar', train=False, download=True,
                           transform=transform, classes=classes)

    elif dataset == 'detection mnist':
        from .mnist import MNIST
        if mode == 'train':
            transform =transforms.Compose([
                transforms.ToTensor()
            ])
            return MNIST(train=True, transform=transform)
        elif mode == 'test':
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            return MNIST(train=False, transform=transform)

    else :
        raise Exception("No dataset named " + dataset)
