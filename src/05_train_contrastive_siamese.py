import os
import glob
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
sys.path.append('/cluster/yinan/cnn_similarity_analysis')

from src.lib.loss import ContrastiveLoss, TripletLoss
from src.lib.siamese.args import  siamese_args
from src.lib.siamese.dataset import generate_extraction_dataset, get_transforms
from src.lib.augmentations import *
from src.data.siamese_dataloader import ContrastiveTrainList
from src.lib.siamese.model import ContrastiveSiameseNetwork


def train(args, augmentations_list):
    if args.device == "cuda:0":
        print("hardware_image_description:", torch.cuda.get_device_name(0))

    # Getting the query, train, ref.. lists
    #get_necessary_lists(args, dataset)
    query_list = [l.strip() for l in open(args.query_list, "r")]
    database_list = [l.strip() for l in open(args.db_list, "r")]
    train_list = [l.strip() for l in open(args.train_list, "r")]
    

    # creating the dataset
    _, _, train_images = generate_extraction_dataset(query_list, database_list, train_list)
    train_list = train_images

    # defining the transforms
    transforms = get_transforms(args)

    # Defining the fixed validation dataloader for modular evaluation
    val_list = train_images[0:args.len]
    val_pairs = ContrastiveTrainList(val_list, train_images, transform=transforms, imsize=args.imsize, argumentation=augmentations_list)
    val_dataloader = DataLoader(dataset=val_pairs, shuffle=True, num_workers=args.num_workers,
                                batch_size=args.batch_size)

    print("loading siamese model")
    net = ContrastiveSiameseNetwork(args.model)
    if not args.start:
        state_dict = torch.load(args.net + args.checkpoint)
        net.load_state_dict(state_dict)
    net.to(args.device)

    # Defining the criteria for training
    criterion = ContrastiveLoss()
    criterion.to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay)

    loss_history = list()
    epoch_losses = list()
    train_losses = list()
    epoch_size = int(len(train_list) / args.num_epochs)
    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        training_subset = train_list[epoch * epoch_size: (epoch + 1) * epoch_size - 1]
        image_pairs = ContrastiveTrainList(training_subset, train_images,
                                           transform=transforms, imsize=args.imsize, argumentation=augmentations_list)
        train_dataloader = DataLoader(dataset=image_pairs, shuffle=True, num_workers=args.num_workers,
                                      batch_size=args.batch_size)

        # Training over batches
        for i, batch in enumerate(train_dataloader, 0):
            query_img, reference_img, label = batch
            query_img = query_img.to(args.device)
            reference_img = reference_img.to(args.device)
            label = label.to(args.device)

            output = net(query_img, reference_img)
            optimizer.zero_grad()
            loss = criterion(output, label, args.margin)
            loss.backward()
            optimizer.step()
            loss_history.append(loss)

        mean_loss = torch.mean(torch.Tensor(loss_history))
        loss_history.clear()

        print("Epoch:{},  Current training loss {}\n".format(epoch, mean_loss))
        train_losses.append(mean_loss)  # Q: Does this only store the mean loss of the last 10 iterations/batches?

        # Validating over batches
        val_loss = []
        with torch.no_grad():
            for j, batch in enumerate(val_dataloader, 0):
                query_img, reference_img, label = batch
                query_img = query_img.to(args.device)
                reference_img = reference_img.to(args.device)
                label = label.to(args.device)

                output = net(query_img, reference_img)
                val_loss.append(criterion(output, label, args.margin))
            val_loss = torch.mean(torch.Tensor(val_loss))
        print("Epoch:{},  Current validation loss {}\n".format(epoch, val_loss))
        epoch_losses.append(val_loss.cpu())

        # This re-write the model if validation loss is lower
        if val_loss.cpu() <= best_val_loss:
            best_val_loss = val_loss.cpu()
            best_model_name = 'Siamese_best.pth'
            model_full_path = args.net + best_model_name
            torch.save(net.state_dict(), model_full_path)
            print('best model updated\n')

        # # This saves model at each epoch - then finds the best model
        # # Replace this by saving only when best model for validation loss, re-write the model
        # trained_model_name = 'Siamese_Epoch_{}.pth'.format(epoch)
        # model_full_path = args.net + trained_model_name
        # torch.save(net.state_dict(), model_full_path)
        # print('model saved as: {}\n'.format(trained_model_name))

    epoch_losses = np.asarray(epoch_losses)
    train_losses = np.asarray(train_losses)
    epochs = np.asarray(range(args.epoch))

    # Loss plot
    plt.title('Loss Visualization')
    plt.plot(epochs, train_losses, color='blue', label='training loss')
    plt.plot(epochs, epoch_losses, color='red', label='validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(args.images + 'loss.png')
    plt.show()

    # # Saving model with best validation loss
    # best_epoch = np.argmin(epoch_losses)
    # best_model_name = 'Siamese_Epoch_{}.pth'.format(best_epoch)
    # pth_files = glob.glob(args.net + '*.pth')
    # pth_files.remove(args.net + best_model_name)
    # for file in pth_files:
    #     os.remove(file)
    # print("best model is: {} with validation loss {}\n".format(best_model_name, epoch_losses[best_epoch]))


if __name__ == "__main__":

    siamese_args = siamese_args()

    # defining augmentations
    augmentations_list = [
        VerticalFlip(probability=0.25),
        HorizontalFlip(probability=0.25),
        AuglyRotate(1.0),
    ]

    train(siamese_args, augmentations_list)
