
def train_model(model, dataloads, scheduler,criterion=None,num_epochs=35,board=None,
                train_steps=None, val_steps=None, model_save='val_loss',use_cuda=False,model_save_path='./runs',start_epoch=0,lr=None):
    # train step 'None' is iter all the datasets ,if num ,will iter num step.
    since = time.time()

    best_model_wts = 0
    best_acc = 10.0
    best_train_loss = 100
    best_val_loss=100
    print('dataset-size:',len(dataloads['train']),len(dataloads['val']))
    # print the para num in the network
    parasum=sum(p.numel() for p in model.parameters())
    print('#### the model para num is : ' , parasum)

    if board:
        board.add_graph(model, (dataloads['train'].dataset[0]['encode_input'].float().unsqueeze(0)))

    if start_epoch!=0:
        load_state=torch.load(os.path.join(model_save_path ,'1.best_model_wts'))
        start_epoch=load_state['epoch']
        state=load_state['state']
        model.load_state_dict(state[i])

    if torch.cuda.device_count() > 1 and use_cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model=nn.DataParallel(model).cuda()

        #load=torch.load('50k.1.best_model_wts')
        #models[0].load_state_dict(load[0])
        print('model is on cuda by gpus!')
    elif torch.cuda.device_count() == 1 and use_cuda:
        model=model.cuda()
        print('model is on cuda!')

    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0

        # ??????????????
        for phase in ['train', 'val']:
            print(phase+' steps is running!')
            steps = 0
            if phase == 'train':
                if train_steps==None:
                    steps=0
                else:
                    steps = train_steps
                # ?? training
                model=model.requires_grad_(True)

            elif phase=='val':
                if val_steps==None:
                    steps=0
                else:
                    steps = val_steps
                # ?? evaluate
                model=model.requires_grad_(False)

            epoch_loss = 0.0
            epoch_acc = 0.0
            stop_setps=0

            # ????
            for i, dataset in enumerate(dataloads[phase]):
                # ????,????????????????????????????key????
                encode_input=dataset['encode_input'].float()
                #encode_mask=dataset['encode_mask']
                encode_target=dataset['encode_target'].float()


                # ?cuda????n m
                if use_cuda:
                    encode_input=encode_input.cuda()
                    #encode_mask=encode_mask.cuda()
                    encode_target=encode_target.cuda()

                # ????
                scheduler.optimizer.zero_grad()

                # ??
                output_v = model(encode_input)
                loss = F.mse_loss(output_v,encode_target)+F.l1_loss(output_v,encode_target)

                # ??????????????,???????????step
                if phase == 'train':
                    loss.backward()
                    scheduler.optimizer.step()

                acc=loss
                # ????loss
                epoch_loss += loss.item()
                epoch_acc += acc.item()

                if sys.version_info.major == 2:
                    sys.stdout.write('{} Loss: {:.4f} Acc: {:.4f}  Step:{:3d}/{:3d} \r'.format(
                        phase, loss.item(), acc.item(), i + 1, steps))
                    sys.stdout.flush()
                elif sys.version_info.major == 3:
                    print('{} Loss: {:.4f} Acc: {:.4f}  Step:{:3d}/{:3d}'.format(
                        phase, loss.item(), acc.item(), i + 1, steps), end='\n')

                if i == steps - 1:
                    break
                stop_setps=i
            print('\n')

            if phase == 'train':
                train_epoch_loss = epoch_loss / stop_setps
                train_epoch_acc = epoch_acc / stop_setps

            # epoch_acc=None
            elif phase == 'val':
                val_epoch_loss = epoch_loss / stop_setps
                val_epoch_acc = epoch_acc / stop_setps

        # ??????????????????????
        if model_save=='train_loss':
            if train_epoch_loss < best_train_loss:
                best_train_loss = train_epoch_loss
                best_epoch=epoch
                if torch.cuda.device_count()>1 and use_cuda:
                    best_model_wts = copy.deepcopy(model.module).cpu()
                else:
                    best_model_wts = copy.deepcopy(model).cpu()
            else:
                print('train loss model has not improving!')
        elif model_save=='val_loss':
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_epoch = epoch
                if torch.cuda.device_count() > 1 and use_cuda:
                    best_model_wts = copy.deepcopy(model.module).cpu()
                else:
                    best_model_wts = copy.deepcopy(model).cpu()
            else:
                print('val loss model has not improving!')

        scheduler.step()
        # ??????????????????????????
        torch.save({'epoch':epoch,'state':best_model_wts,'comments':'u-net2'}, os.path.join(model_save_path ,'7.best_model_wts'))

        ## ?????????????????
        if board:
            if torch.cuda.device_count() > 1 and use_cuda:
                for name, para in model.module.named_parameters():
                    try:
                        board.add_histogram(name, para.clone().cpu().data.numpy(), epoch)
                    except:
                        board.add_histogram(name,np.zeros(para.shape),epoch)
            else:
                for name, para in model.named_parameters():
                    try:
                        board.add_histogram(name, para.clone().cpu().data.numpy(), epoch)
                    except:
                        board.add_histogram(name,np.zeros(para.shape),epoch)

        print('train: Loss: {:.4f} Acc: {:.4f} ;  val: Loss: {:.4f} Acc: {:.4f}'.format(
            train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc))
        print('\n')
        if board:
            try:
                board.add_scalars('all_', {'train_loss': train_epoch_loss, 'train_acc': train_epoch_acc,'val_loss': val_epoch_loss, 'val_acc': val_epoch_acc}, epoch)
            except:
                continue

    hpara = {'batch_size_train': dataloads['train'].batch_size,
             'batch_size_val': dataloads['val'].batch_size, 'window_size': dataloads['train'].dataset.window_size,
             'train_size': len(dataloads['train'].dataset), 'val_size': len(dataloads['val'].dataset),'lr':lr,'best_epoch':best_epoch,'num_para':parasum,'missing_rate':0.15}
    mpara={'best_train_loss': best_train_loss, 'best_val_loss': best_val_loss}
    board.add_hparams(hparam_dict=hpara,metric_dict=mpara)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best train Loss: {:4f}'.format(best_train_loss))
    print('Best val Loss: {:4f}'.format(best_val_loss))
    if board:
        board.close()
