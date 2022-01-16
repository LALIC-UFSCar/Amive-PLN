import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import numpy as np
from transformers import BertForSequenceClassification,BertTokenizer

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


#passar modelo
state = torch.load(map_location=device, f= "tweetsentmodel.pt" )
model = BertForSequenceClassification.from_pretrained('resources/bert-base-portuguese-cased')
model.load_state_dict(state_dict= state, strict=False)
tokenizer = BertTokenizer.from_pretrained('resources/vocab.txt')




def getSentiment(sentences):
  input_ids = []
  attention_masks = []
  for sent in sentences:   
      # `encode_plus` will:
      #   (1) Tokenize the sentence.
      #   (2) Prepend the `[CLS]` token to the start.
      #   (3) Append the `[SEP]` token to the end.
      #   (4) Map tokens to their IDs.
      #   (5) Pad or truncate the sentence to `max_length`
      #   (6) Create attention masks for [PAD] tokens.
      encoded_dict = tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 280,           # Pad & truncate all sentences.
                          padding= 'max_length',
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',
                          truncation = True     # Return pytorch tensors.
                     )
      
      # Add the encoded sentence to the list.    
      input_ids.append(encoded_dict['input_ids'])
      
      # And its attention mask (simply differentiates padding from non-padding).
      attention_masks.append(encoded_dict['attention_mask'])
  
    # Convert the lists into tensors.
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  
    # Set the batch size.  
  batch_size = 64
  
    # Create the DataLoader.
  prediction_data = TensorDataset(input_ids, attention_masks)
  prediction_sampler = SequentialSampler(prediction_data)
  prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


  #TODO: colocar um if aqui caso o ambiente de execução não tenha GPU, ao invés de comentar a linha abaixo
  model.cuda()
  # Put model in evaluation mode
  model.eval()

  # Tracking variables 
  predictions = []

  # Predict 
  for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask = batch

    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, 
                        attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # Store predictions and true labels
    predictions.append(logits)
  aux = []
  for i in range(len(predictions)):
    aux = np.argmax(predictions[i], axis=1).flatten()
  return aux

