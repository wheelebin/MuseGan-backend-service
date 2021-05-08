def run():
    print("I just ran again!")
while True:
    run_again = input("Run again?: [y/n] ")
    if run_again.lower() != "y":
        break
    print("Will run again")
    try:
        run()
    except AssertionError as error:
        print(error)
        print("An exception has occured")
        pass

