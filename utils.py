import gc
import os
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Optional, Iterator, Tuple
import numpy as np
from pathlib import Path
import re
# import mmap

    # def text_generator(self, file_path: str) -> Iterator[str]:
    #     with open(file_path, 'rb') as f:
    #         with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
    #             accumulated_text = b""
    #             boundary_markers = [b'.', b'!', b'?', b'\n\n']

    #             while True:
    #                 chunk = mm.read(self.chunk_size)
    #                 if not chunk and not accumulated_text:
    #                     break

    #                 current_text = accumulated_text + chunk
    #                 last_boundary = -1

    #                 for marker in boundary_markers:
    #                     pos = current_text.rfind(marker)
    #                     if pos > last_boundary and len(current_text) >20:
    #                         last_boundary = pos

    #                 if last_boundary != -1:
    #                     text_to_process = current_text[:last_boundary + 1]
    #                     accumulated_text = current_text[last_boundary + 1:]
    #                 else:
    #                     text_to_process = current_text[:-self.overlap_size]
    #                     accumulated_text = current_text[-self.overlap_size:]

    #                 try:
    #                     decoded_text = text_to_process.decode('utf-8', 'ignore').replace(':', ' ')
    #                     if decoded_text.strip():
    #                         yield decoded_text
    #                 except UnicodeDecodeError:
    #                     pass

    #                 if len(accumulated_text) > self.max_chunk_buffer_size:
    #                     try:
    #                         decoded_leftover = accumulated_text.decode('utf-8', 'ignore').replace(':', ' ')
    #                         if decoded_leftover.strip():
    #                             yield decoded_leftover
    #                     except UnicodeDecodeError:
    #                         pass
    #                     accumulated_text = b""

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class StreamingDataset(IterableDataset):
    def __init__(
        self,
        files: List[str],
        tokenizer,
        sequence_length: int = 512,
        chunk_size: int = 30 * 1024 * 1024,  # 30MB chunks
        stride: int = 256,
        batch_buffer_size: int = 5000,
        shuffle_buffer: bool = True,
        overlap_size: int = 1024,
        max_chunk_buffer_size: int = 1000*1024*1024, # 2GB
    ):
        super().__init__()
        self.files = files
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.stride = stride
        self.batch_buffer_size = batch_buffer_size
        self.shuffle_buffer = shuffle_buffer
        self.overlap_size=overlap_size
        self.max_chunk_buffer_size = max_chunk_buffer_size
        self.boundary_markers = [b'.', b'!', b'?', b'\n']
        self.eos_token_id=3
        self.sos_token_id =2

    def text_generator(self, file_path: str) -> Iterator[str]:
        with open(file_path, 'rb') as f:
                
                accumulated_text = b""
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk and not accumulated_text:
                        break


                    current_text = accumulated_text + chunk
                    last_boundary = -1

                    if len(chunk) < self.chunk_size:
                        try:
                            decoded_text = current_text.decode('utf-8', 'ignore').replace(':', " ").strip()
                            if decoded_text:
                                yield decoded_text
                        except UnicodeDecodeError:
                            pass

                        break

                    for marker in self.boundary_markers:
                        pos = current_text.rfind(marker)
                        if pos > last_boundary:
                            last_boundary = pos

                    if last_boundary != -1:
                        text_to_process = current_text[:last_boundary + 1]
                        accumulated_text = current_text[last_boundary + 1:]
                    else:
                        text_to_process = current_text[:-self.overlap_size]
                        accumulated_text = current_text[-self.overlap_size:]

                    try:
                        decoded_text = text_to_process.decode('utf-8', 'ignore').replace(':'," ").strip() # the data has so much : 
                        if decoded_text:
                            yield decoded_text
                    except UnicodeDecodeError:
                        pass

                    if len(accumulated_text) > self.max_chunk_buffer_size:
                        try:
                            decoded_leftover = accumulated_text.decode('utf-8', 'ignore').replace(':'," ").strip()
                            if decoded_leftover:
                                yield decoded_leftover
                        except UnicodeDecodeError:
                            pass
                        accumulated_text = b""

    def process_file(self, file_path: str):
        # tokens = [self.sos_token_id]
        tokens = []
        batch = []

        for chunk in self.text_generator(file_path):
            sentences = re.findall(r'[^.\n]+[.\n]+', chunk)
            tokenized_chunks = self.tokenizer.encode_batch(sentences, add_special_tokens=False)

            for tok_ids in tokenized_chunks:
                tokens.extend(tok_ids.ids)

                while len(tokens) >= (self.sequence_length + 1):
                    input_seq = tokens[:self.sequence_length]
                    target_seq = tokens[1:self.sequence_length + 1]
                    batch.append((torch.tensor(input_seq), torch.tensor(target_seq)))
                    tokens = tokens[self.stride:]

                    if len(batch) >= self.batch_buffer_size:
                        if self.shuffle_buffer:
                            np.random.shuffle(batch)
                        yield from batch
                        batch = []

            
        # tokens.append(self.eos_token_id)

        # if len(tokens) == (self.sequence_length + 1):
        #     input_seq = tokens[:self.sequence_length]
        #     target_seq = tokens[1:self.sequence_length + 1]
        #     batch.append((torch.tensor(input_seq), torch.tensor(target_seq)))

        if len(tokens) >= int(self.sequence_length * 0.8):
            padding_length = self.sequence_length - len(tokens)
            input_seq = tokens + [0] * padding_length
            target_seq = tokens[1:] + [0] * (padding_length + 1)
            batch.append((torch.tensor(input_seq), torch.tensor(target_seq)))

        

        if batch:
            if self.shuffle_buffer:
                np.random.shuffle(batch)
            yield from batch



    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            files_to_process = self.files
        else:
            per_worker = int(np.ceil(len(self.files) / worker_info.num_workers))
            start_idx = worker_info.id * per_worker
            end_idx = min(start_idx + per_worker, len(self.files))
            files_to_process = self.files[start_idx:end_idx]

        try:
            for file_path in files_to_process:
                try:
                    for batch in self.process_file(file_path):
                        yield batch
                    
                    # Force garbage collection after each file
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # If using GPU
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
        finally:
            # Clean up at the end of iteration
            gc.collect()
            torch.cuda.empty_cache()

def create_streaming_dataloaders(
    files: List[str],
    tokenizer,
    batch_size: int = 32,
    sequence_length: int = 512,
    stride: int = 256,
    batch_buffer_size: int = 1000,
    num_workers: Optional[int] = None,
    shuffle_buffer: bool = True,
) -> DataLoader:
    """Create dataloader with streaming dataset."""
    if num_workers is None:
        num_workers = min(len(files), torch.multiprocessing.cpu_count() - 4)
    
    dataset = StreamingDataset(
        files=files,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        stride=stride,
        batch_buffer_size=batch_buffer_size,
        shuffle_buffer=shuffle_buffer
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # pin_memory=True,
        # prefetch_factor= 2 if  num_workers > 0 else None,
        persistent_workers=True if  num_workers > 0 else False
    )

def train(model, embedding, optimizer, scheduler, epoch, trainloader, validloader, criterion, num_layers, use_amp=True, clip =None, prefix=''):
    embedding = embedding.to(device)
    model = model.to(device)
    
    least_loss = float('inf')
    train_losses, valid_losses = [], []
    
    # Add gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    try:
        for e in range(epoch):
            model.train()
            total_num = 0
            train_loss = 0
            
            # Use tqdm with position=0 to prevent multiple progress bars
            pbar = tqdm(trainloader, position=0, leave=True)
            
            for batch_idx, (data, labels) in enumerate(pbar):
                
                try:
                    data = data.to(device)
                    labels = labels.to(device)
                    
                    if use_amp:
                        # Use automatic mixed precision
                        with torch.amp.autocast('cuda'):
                            h = model(embedding(data))
                            predictions = torch.nn.functional.linear(h, embedding.weight)
                            loss = criterion(predictions.view(-1, predictions.shape[-1]), labels.view(-1))
                        if loss.isnan():
                            print(f"skip: batch {batch_idx} for nan")
                            continue

                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        if clip:
                            scaler.unscale_(optimizer)
                            parameters = list(model.parameters()) + list(embedding.parameters())
                            torch.nn.utils.clip_grad_norm_(parameters, max_norm=clip)


                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Regular training
                        h = model(embedding(data))
                        predictions = torch.nn.functional.linear(h, embedding.weight)
                        loss = criterion(predictions.view(-1, predictions.shape[-1]), labels.view(-1))
                        if loss.isnan():
                            print("skip: batch {batch_idx} for nan")
                            continue

                        optimizer.zero_grad(set_to_none=True)
                        if clip:
                            parameters = list(model.parameters()) + list(embedding.parameters())
                            torch.nn.utils.clip_grad_norm_(parameters, max_norm=clip)
                        loss.backward()
                        optimizer.step()
                    
                    # Update metrics
                    train_loss += loss.item()
                    total_num += 1
                    
                    # Update progress bar
                    if batch_idx % 100 == 0:
                        pbar.set_description(f'Epoch {e+1} Loss: {loss.item():.4f}')
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    

                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: out of memory at batch {batch_idx}. Skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"ERROR:{e} Skipping batch {batch_idx}")
                        continue
            
            # Calculate average training loss
            train_loss = train_loss / total_num if total_num > 0 else float('inf')
            
            # Validation phase
            model.eval()
            valid_loss = evaluate(model, embedding, validloader, criterion, use_amp)
            
            # Save checkpoints
            if least_loss > valid_loss:
                least_loss = valid_loss
                save_checkpoint({
                    'epoch': e + 1,
                    'model_state_dict': model.state_dict(),
                    'embedding_state_dict': embedding.state_dict(),
                    'loss': valid_loss,
                }, is_best=True, prefix=prefix, num_layers=num_layers)
            
            # Save regular checkpoint
            save_checkpoint({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'embedding_state_dict': embedding.state_dict(),
                'loss': valid_loss,
            }, is_best=False, prefix=prefix, num_layers=num_layers)
            
            print(f'Epoch:{e+1} | train loss:{train_loss:.4f} | valid loss:{valid_loss:.4f} | lr: {scheduler.get_last_lr()}')
            print('-' * 100)
            
            scheduler.step()
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            
    except Exception as error:
        print(f"Training interrupted: {str(error)}")
        # Save emergency checkpoint
        save_checkpoint({
            'epoch': e + 1,
            'model_state_dict': model.state_dict(),
            'embedding_state_dict': embedding.state_dict(),
            'loss': valid_loss if 'valid_loss' in locals() else float('inf'),
        }, is_best=False, prefix=prefix + 'emergency_', num_layers=num_layers)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        
    return train_losses, valid_losses

def save_checkpoint(state, is_best, prefix='', num_layers=0):
    filename = f'checkpoints/{prefix}{"best_" if is_best else "last_"}'
    torch.save(state, f'{filename}checkpoint{num_layers}.pt')
    
def evaluate(model, embedding, validloader, criterion,  use_amp=True):
    model.eval()
    with torch.no_grad():
        total_num = 0
        valid_loss = 0
        for data, labels in tqdm(validloader, desc='Validating'):
            try:
                data = data.to(device)
                labels = labels.to(device)
                if use_amp:
                    # Use automatic mixed precision
                    with torch.amp.autocast('cuda'):
                        h = model(embedding(data))
                        predictions = torch.nn.functional.linear(h, embedding.weight)
                        loss = criterion(predictions.view(-1, predictions.shape[-1]), labels.view(-1))   
                else:
                    # Regular training
                    h = model(embedding(data))
                    predictions = torch.nn.functional.linear(h, embedding.weight)
                    loss = criterion(predictions.view(-1, predictions.shape[-1]), labels.view(-1))

                valid_loss += loss.item()
                total_num += labels.size(0)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: out of memory at batch ++{total_num}. Skipping batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"ERROR:{e} Skipping batch ++{total_num}")
                    continue
                    
    return valid_loss / total_num if total_num > 0 else float('inf')

def load_model_and_embeddings(
    model: torch.nn.Module,
    embedding: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = False
):
    """
    Load model and embeddings from a checkpoint file.
    
    Args:
        model_class: Uninitialized model class
        embedding_class: Uninitialized embedding class
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
        strict: Whether to strictly enforce state dict loading
        
    Returns:
        Tuple of (model, embeddings, checkpoint_info)
    """
    try:
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        
        # Load state dicts
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        except Exception as e:
            print(f"Warning: Error loading model state dict: {e}")
            if strict:
                raise
                
        try:
            embedding.load_state_dict(checkpoint['embedding_state_dict'], strict=strict)
        except Exception as e:
            print(f"Warning: Error loading embedding state dict: {e}")
            if strict:
                raise
        
        # Extract training info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', float('inf')),
        }
        
        print(f"Successfully loaded checkpoint from epoch {checkpoint_info['epoch']}")
        return model, embedding, checkpoint_info
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def find_latest_checkpoint(
    checkpoint_dir: str,
    num_layers: int,
    prefix: str = '',
    prefer_best: bool = False
) -> Optional[str]:
    """
    Find the latest checkpoint file in the given directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        num_layers: Number of layers in the model
        prefix: Prefix used in checkpoint naming
        prefer_best: Whether to prefer 'best' checkpoints over 'last' ones
        
    Returns:
        Path to the latest checkpoint file or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Define checkpoint patterns to search for
    patterns = []
    if prefer_best:
        patterns.extend([
            f"{prefix}best_checkpoint{num_layers}.pt",
            f"{prefix}last_checkpoint{num_layers}.pt",
            f"{prefix}emergency_checkpoint{num_layers}.pt"
        ])
    else:
        patterns.extend([
            f"{prefix}last_checkpoint{num_layers}.pt",
            f"{prefix}best_checkpoint{num_layers}.pt",
            f"{prefix}emergency_checkpoint{num_layers}.pt"
        ])
    
    # Search for checkpoint files
    for pattern in patterns:
        checkpoint_path = checkpoint_dir / pattern
        if checkpoint_path.exists():
            return str(checkpoint_path)
    
    return None

def setup_training_state(
    model: torch.nn.Module,
    embedding: torch.nn.Module,
    checkpoint_dir: str,
    num_layers: int,
    prefix: str = '',
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[torch.nn.Module, torch.nn.Module, int]:
    """
    Set up complete training state, either fresh or from checkpoint.
    
    Args:
        model: Model class
        embedding: Embedding class
        optimizer_class: Optimizer class
        scheduler_class: Scheduler class
        checkpoint_dir: Directory containing checkpoints
        num_layers: Number of layers in model
        learning_rate: Learning rate for optimizer
        prefix: Prefix for checkpoint files
        device: Device to load model to
        
    Returns:
        Tuple of (model, embedding, optimizer, scheduler, start_epoch)
    """
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint(checkpoint_dir, num_layers, prefix)
    start_epoch = 0
    
    if checkpoint_path:
        try:
            # Load model and embedding from checkpoint
            model, embedding, checkpoint_info = load_model_and_embeddings(
                model, embedding, checkpoint_path, device
            )
        
            

                
            start_epoch = checkpoint_info['epoch']
            print(f"Resumed training from epoch {start_epoch}")
            
        except Exception as e:
            print(f"Error loading checkpoint, starting fresh: {e}")
            model = model.to(device)
            embedding = embedding.to(device)
            
    else:
        print("No checkpoint found, starting fresh training")
        model = model.to(device)
        embedding = embedding.to(device)

        
    return model, embedding, start_epoch

def calculate_perplexity(model, embedding, dataloader, use_amp=True, device='cuda'):
    embedding = embedding.to(device)
    model = model.to(device)
    
    # Set models to evaluation mode
    embedding.eval()
    model.eval()
    
    total_loss = 0
    total_words = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for data, targets in dataloader:
            # Move data to device
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass through embedding and LSTM
            if use_amp:
                # Use automatic mixed precision
                with torch.amp.autocast('cuda'):
                    h = model(embedding(data))
                    logits = torch.nn.functional.linear(h, embedding.weight)
                    # Calculate loss for each word
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))


            else:
                # Regular 
                h = model(embedding(data))
                logits = torch.nn.functional.linear(h, embedding.weight)
                # Calculate loss for each word
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            

            total_loss += loss.item()
            total_words += targets.numel()
    
    # Calculate average negative log likelihood
    avg_nll = total_loss / total_words

    
    # Calculate perplexity
    perplexity = np.exp(avg_nll)
    
    return perplexity

def sample_from_model(model, embedding,tokenizer,start_sequence, top_k=20, num_generate=4.,use_amp=True ,device ='cuda'):
    model.eval()  # Set the model to evaluation mode
    embedding.eval()
    model = model.to(device)
    embedding = embedding.to(device)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    
    # Convert start sequence to indices
    tokens = tokenizer.encode(start_sequence, add_special_tokens=False).ids
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)  # Shape: [1, seq_len]
    
    with torch.no_grad():  # No need to track gradients
        for _ in range(num_generate):

                        # Forward pass through embedding and LSTM
            if use_amp:
                # Use automatic mixed precision
                with torch.amp.autocast('cuda'):
                    h = model(embedding(input_ids))
                    predictions = torch.nn.functional.linear(h, embedding.weight)
            else:
                # Regular training
                h = model(embedding(input_ids))
                predictions = torch.nn.functional.linear(h, embedding.weight)
            
            # Get top k logits and indices for the last token
            values, indices = torch.topk(predictions[0, -1], top_k)  # Shape: [top_k]
            
            # Apply softmax to convert logits to probabilities
            probs = torch.nn.functional.softmax(values, dim=0)  # Shape: [top_k]
            
            # Sample from the probabilities
            idx = torch.multinomial(probs, num_samples=1)  # Shape: [1]
            next_token_id = indices[idx]  # Shape: [1]

            # if next_token_id.item() == last_token:
            #     break
                
            # Reshape next_token_id to match input_ids dimensions [1, 1]
            next_token_id = next_token_id.view(1, 1)
                
            # Append to input for next round
            input_ids = torch.cat([input_ids, next_token_id], dim=1)  # Shape: [1, seq_len + 1]
    
    # Convert to numpy array before decoding
    return tokenizer.decode_batch(input_ids.cpu().detach().numpy())