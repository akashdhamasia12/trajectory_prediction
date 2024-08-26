    print('Training')
    model.train()
    train_running_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    counter = 0
    uncertainty_var = 0
    eauc_loss = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        #print(i)
        counter += 1
        image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        availability = availability.view(availability.size(0), -1)
        optimizer.zero_grad()

        #denormalization
        keypoints = ((keypoints + 1)/2)*int(config.IMAGE_SIZE)            

        for mc in range(0, config.num_monte_carlo_training):
            outputs, _ = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping    
            #denormalization
            outputs = ((outputs + 1)/2)*int(config.IMAGE_SIZE)            
            outputs = outputs.view(keypoints.size(0), -1)
            outputs = torch.unsqueeze(outputs, 0)
            if mc == 0:
                outputs_list = outputs
            else:
                outputs_list = torch.cat((outputs_list, outputs), 0)

        # outputs = torch.mean(outputs_list, 0)
        outputs = torch.squeeze(outputs_list[0], 0)

        if config.num_monte_carlo_training > 1:
            #denormalisation
            # outputs_list = ((outputs_list + 1)/2)*int(config.IMAGE_SIZE)            
            outputs_var = torch.std(outputs_list, 0)
            outputs_var_ = outputs_var.view(outputs_var.size(0), -1, 2)
            outputs_var_mean = torch.mean(outputs_var_, 2)
            total_uncertainty_ = torch.mean(outputs_var_mean, 1)
            total_uncertainty = torch.mean(total_uncertainty_)

        if config.LOSS_FUNCTION == "ADAPTIVE":
            loss = torch.mean(adaptive.lossfun((outputs - keypoints))) #[:,None] # (y_i - y)[:, None] # numpy array or tensor
        elif config.LOSS_FUNCTION == "MSE":
            loss = criterion(outputs, keypoints)

        if config.quantile_regression == True:
            loss = torch.max(config.quantile*loss, (config.quantile-1)*loss)

        if config.train_evaluate:
            ade, fde = evaluate(outputs.clone(), keypoints.clone(), availability.clone())

        if config.num_monte_carlo_training > 1:
        #     loss = loss + outputs_var*config.uncertainty_factor
            EaUC = get_EAUC_loss(torch.mean(loss, 1), total_uncertainty_)
            # print(EaUC)

        
        loss = loss * availability        
        loss = loss.mean()

        if config.num_monte_carlo_training > 1:
            loss = loss + EaUC

        train_running_loss += loss.item()
        total_ade += ade.item()
        total_fde += fde.item()
        if config.num_monte_carlo_training > 1:
            uncertainty_var += total_uncertainty.item()    
            eauc_loss += EaUC.item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss/counter
    train_ade = total_ade/counter
    train_fde = total_fde/counter
    train_uncertainty = uncertainty_var/counter
    train_eauc = eauc_loss/counter

    return train_loss, train_ade, train_fde, train_uncertainty, train_eauc
