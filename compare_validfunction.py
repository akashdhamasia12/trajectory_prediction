    print('Validating')
    # model.eval()
    valid_running_loss = 0.0
    total_ade_ = 0.0
    total_fde_ = 0.0
    uncertainty_var = 0
    eauc_loss = 0


    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            # outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
            # outputs = outputs.view(keypoints.size(0), -1)
            availability = availability.view(availability.size(0), -1)

            #denormalization
            keypoints = ((keypoints + 1)/2)*int(config.IMAGE_SIZE)            

            for mc in range(0, config.num_monte_carlo_training):
                model.eval()
                model = set_training_mode_for_dropout(model, True)
                outputs, _ = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping    
                #denormalization
                outputs = ((outputs + 1)/2)*int(config.IMAGE_SIZE)            
                outputs = outputs.view(keypoints.size(0), -1)
                outputs = torch.unsqueeze(outputs, 0)
                if mc == 0:
                    outputs_list = outputs
                else:
                    outputs_list = torch.cat((outputs_list, outputs), 0)
                model = set_training_mode_for_dropout(model, False)

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

            loss = loss * availability
            loss = loss.mean()

            if config.num_monte_carlo_training > 1:
                loss = loss + EaUC

            valid_running_loss += loss.item()
            total_ade_ += ade.item()
            total_fde_ += fde.item()
            if config.num_monte_carlo_training > 1:
                uncertainty_var += total_uncertainty.item()    
                eauc_loss += EaUC.item()

    valid_loss = valid_running_loss/counter
    val_ade = total_ade_/counter
    val_fde = total_fde_/counter
    val_uncertainty = uncertainty_var/counter
    val_eauc = eauc_loss/counter

    return valid_loss, val_ade, val_fde, val_uncertainty, val_eauc
