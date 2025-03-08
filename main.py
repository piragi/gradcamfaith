import pipeline as pipe

def main():
    image = "./images/xray.jpg"
    pipe.perturb_classify_all_patches(image)


if __name__ == "__main__":
    main()
