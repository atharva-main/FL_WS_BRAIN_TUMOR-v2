from cryptography.fernet import Fernet

def key_generate():
    key = Fernet.generate_key()
    print(f"Key Generated: {key}")
    cipher_suite = Fernet(key)
    return cipher_suite



def encrypt_obj(cipher_suite,obj):

    return cipher_suite.encrypt(obj)


def decrypt_obj(cipher_suite,encrypted_data):

    return cipher_suite.decrypt(encrypted_data)
