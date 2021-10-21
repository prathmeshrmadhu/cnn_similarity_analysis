import os
import glob
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
sys.path.append('/cluster/yinan/cnn_similarity_analysis/')

from src.lib.loss import TripletLoss
from src.lib.siamese.args import  siamese_args
from src.lib.siamese.dataset import generate_train_dataset, get_transforms
from src.lib.augmentations import *
from src.data.siamese_dataloader import TripletTrainList, TripletValList
from src.lib.siamese.model import TripletSiameseNetwork


def train(args, augmentations_list):
    if args.device == "gpu":
        print("hardware_image_description:", torch.cuda.get_device_name(0))

    query_ind = list(range(1000, 1300))
    query = [str(l) + '00' for l in query_ind]
    ref_positive = [str(l) + '01' for l in query_ind]
    ref_negative = []
    for i in range(len(query_ind)):
        b = query_ind.copy()
        b.remove(query_ind[i])
        ref_negative.append(str(random.choice(b)) + '00')

    # creating the dataset
    query_images, positive_images, negative_images = generate_train_dataset(query, ref_positive, ref_negative)
    query_train = query_images[0:250]
    p_train = positive_images[0:250]
    n_train = negative_images[0:250]
    train_list = []
    for i in range(len(query_train)):
        train_list.append((query_train[i], p_train[i], n_train[i]))

    # defining the transforms
    transforms = get_transforms(args)

    # Defining the fixed validation dataloader for modular evaluation
    query_val = query_images[250:300]
    p_val = positive_images[250:300]
    n_val = negative_images[250:300]
    val_list = []
    for j in range(len(query_val)):
        val_list.append((query_val[j], p_val[j], n_val[j]))

    val_pairs = TripletValList(val_list, transform=transforms, imsize=args.imsize, argumentation=augmentations_list)
    val_dataloader = DataLoader(dataset=val_pairs, shuffle=True, num_workers=args.num_workers,
                                batch_size=args.batch_size)

    print("loading siamese model")
    net = TripletSiameseNetwork(args.model)
    if not args.start:
        state_dict = torch.load(args.net + args.checkpoint)
        net.load_state_dict(state_dict)
    net.to(args.device)

    # Defining the criteria for training
    criterion = TripletLoss()
    criterion.to(args.device)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
    #                              lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam([{'params': net.head.parameters(), 'lr': args.lr * 0.05},
    #                               {'params': net.fc1.parameters(), 'lr': args.lr},
    #                               {'params': net.fc2.parameters(), 'lr': args.lr}],
    #                              lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    loss_history = list()
    epoch_losses = list()
    train_losses = list()
    # epoch_size = int(len(train_list) / args.epoch)
    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        image_pairs = TripletValList(train_list, transform=transforms, imsize=args.imsize, argumentation=augmentations_list)
        train_dataloader = DataLoader(dataset=image_pairs, shuffle=True, num_workers=args.num_workers,
                                      batch_size=args.batch_size)
        net.train()
        # Training over batches
        for i, batch in enumerate(train_dataloader, 0):
            query_img, rp_img, rn_img = batch
            query_img = query_img.to(args.device)
            rp_img = rp_img.to(args.device)
            rn_img = rn_img.to(args.device)

            p_score, n_score = net(query_img, rp_img, rn_img)
            optimizer.zero_grad()
            loss = criterion(p_score, n_score, args.margin)
            loss.backward()
            optimizer.step()
            loss_history.append(loss)

        mean_loss = torch.mean(torch.Tensor(loss_history))
        loss_history.clear()

        print("Epoch:{},  Current training loss {}\n".format(epoch, mean_loss))
        train_losses.append(mean_loss)  # Q: Does this only store the mean loss of the last 10 iterations?

        # Validating over batches
        val_loss = []
        p_score_list = []
        n_score_list = []
        net.eval()
        with torch.no_grad():
            for j, batch in enumerate(val_dataloader, 0):
                query_img, rp_img, rn_img = batch
                query_img = query_img.to(args.device)
                rp_img = rp_img.to(args.device)
                rn_img = rn_img.to(args.device)

                p_score, n_score = net(query_img, rp_img, rn_img)
                val_loss.append(criterion(p_score, n_score, args.margin))
                p_score_list.append(torch.mean(p_score))
                n_score_list.append(torch.mean(n_score))
            val_loss = torch.mean(torch.Tensor(val_loss))
        print("Epoch:{},  Current validation loss {}\n".format(epoch, val_loss))
        epoch_losses.append(val_loss.cpu())

        # This re-write the model if validation loss is lower
        if val_loss.cpu() <= best_val_loss:
            best_val_loss = val_loss.cpu()
            avg_p_score = torch.mean(torch.Tensor(p_score_list)).cpu()
            avg_n_score = torch.mean(torch.Tensor(n_score_list)).cup()
            best_model_name = 'Triplet_best.pth'
            model_full_path = args.net + best_model_name
            torch.save(net.state_dict(), model_full_path)
            print('best model updated\n')

        # trained_model_name = 'Siamese_Epoch_{}.pth'.format(epoch)
        # model_full_path = args.net + trained_model_name
        # torch.save(net.state_dict(), model_full_path)
        # print('model saved as: {}\n'.format(trained_model_name))
    print("Training finished")
    print("Average Positive Score: {}\n".format(avg_p_score))
    print("Average Negative Score: {}\n".format(avg_n_score))
    epoch_losses = np.asarray(epoch_losses)
    train_losses = np.asarray(train_losses)
    epochs = np.asarray(range(args.num_epochs))

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
