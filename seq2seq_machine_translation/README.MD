A Seq2Seq practice for machine translation 

- **Data Set:** seq2seq_machine_translation/data/
  - "small_vocab_en": the English corpus.
  - "small_vocab_fr": the French corpus.
- **Basic Seq2Seq model:** seq2seq_machine_translation/Seq2seq_char.ipynb
- **Simple "English to French translation" by Seq2Seq:** seq2seq_machine_translation/machine_translation_seq2seq.ipynb
  - Encoder:  Input->word embedding->LSTM->Context Vector
  - Decoder:  Context Vector->target word embedding->LSTM->Traning Output/Prediction Output
- **"English to French" Seq2Seq machine translation by Bi-LSTM and Attention:** seq2seq_machine_translation/mt_attention_birnn.ipynb
  - Encoder(Attention): 
    - 1 input->word embedding->Bi-LSTM-> weight a
    - 2 weight a , s_prev -> concat(weight a, s_prev)->softmax-> new weight a'
    - 3 weight a, new weight a' -> ∑(a*a') -> context Vector
  - Decoder: Context Vector->target word embedding->LSTM->Traning Output/Prediction Output
                             
Details in https://www.jianshu.com/p/b07729236d7f, i will keep updating in English version when i have time.
