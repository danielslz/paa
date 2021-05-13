import re
from time import time


# le conteudo arquivo
def retorna_conteudo_arquivo(caminho_arquivo: str):
    with open(caminho_arquivo) as f:
        return f.read()


# retorna lista de palavras sem caracteres indesejados
def limpa_texto(texto: str):
    # ignora colchete, chave, numero, virgula, ponto, palavra com apenas uma letra
    return re.findall(r'[^\d\W]{2,}', texto, re.UNICODE)


# gera dicionario de anagramas
def gera_lista_anagramas(lista_palavras: list):
    # dicionario de anagramas encontrados
    anagramas = {}

    # percorre lista de palavras
    for palavra in lista_palavras:
        # converte palavra para minusculo
        palavra = palavra.lower()

        ## transforma
        # ordena caracteres da palavra para gerar chave do dicionario
        chave = ''.join(sorted(palavra))

        # checa se dicionario tem chave, se nao tiver adiciona
        if chave not in anagramas.keys():
            anagramas[chave] = set() 
    
        ## conquista
        # adiciona palavra na lista de valores por chave
        # palavras na mesma chave sao anagramas
        anagramas[chave].add(palavra)  

    # retorna dicionario de anagramas
    return anagramas


# imprime valores das chaves do dicionario com mais de uma ocorrencia
def imprime_lista_anagramas(dicionario_anagramas: dict):
    print('=> Resultado:')
    for anagrama in dicionario_anagramas.values():
        if len(anagrama) > 1:
            print(anagrama)


# le texto do arquivo
# wget -q --show-progress -O anagrama.txt https://drive.google.com/uc?id=1b1JXXiiFsK0H6Zt-bfOmEuKdxFSy7ey2
texto = retorna_conteudo_arquivo('anagrama.txt')

# limpa caracteres indesejados
palavras = limpa_texto(texto)

print('=> Palavras a processar: {}'.format(len(palavras)))

# calcula anagramas
inicio = time()
anagramas = gera_lista_anagramas(palavras)
fim = time()

# imprime resultado
imprime_lista_anagramas(anagramas)

print('=> Tempo de execução: {} segundos'.format(fim - inicio))
