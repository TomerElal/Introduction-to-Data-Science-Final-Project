import pandas as pd

users = [
    "elad-itzkovitch-1b29766a",
    "hilla-regev",
    "thetobioluwole",
    "erangefen",
    "adi-soffer-teeni-3b754635",
    "dvir-ross",
    "shirley-kleeblatt",
    "sunchitdudeja",
    "omri-shooshan-67a091147",
    "tali-libman-95781427",
    "dudi-malits-59823531",
    "sudoyasir",
    "jason-salt",
    "joshagardner",
    "ofer-shaltiel-28423261",
    "alon-lee-green-0a48ba141",
    "stav-ziv",
    "dev-g-5949a031b",
    "shirbarnea",
    "lotem-aviv",
    "idit-bar-iditaravit-arab-business-culture",
    "ofir-zuk",
    "yoseph-haddad-a97a47198",
    "liad-krimberg-3b735b1a",
    "shay-kallach-97a8b5146",
    "noa-elharrar",
    "eddie-kanevskie-9249ba54",
    "moshe-radman-abutbul-משה-רדמן-אבוטבול-🇮🇱-8a628260",
    "shlomo-strauss-78a9a6277",
    "oded-weiss-recruitment-for-startups",
    "malka-friedman",
    "danmian",
    "johnrushx",
    "danieldayan",
    "na-ama-schultz-547027101",
    "michal-barkai-brody-a325842a",
    "yael-zeevi-kalomiti",
    "miriam-dissen",
    "jamesfobrien",
    "jonathan-shavit-615bb0215",
    "hasolidit",
    "ram-kedem",
    "merav-pirak",
    "mirimendelson",
    "noatishby",
    "shahar-polak",
    "carlos-arguelles-6352392",
    "naftalibennett",
    "eden-bibas",
    "buchman",
    "eriklidman",
    "maorfaridphd",
    "hilaoren1",
    "oren-helman-792657127",
    "drorgloberman",
    "emilrozenblat",
    "sitvanit-kustka",
    "danielle-biton",
    "amir-barkol",
    "vladismarkin",
    "blaise-bevilacqua",
    "oksana-matiiash-62542b95",
    "milos-jokic",
    "anubhav-khanna2386",
    "evgeny-sinay-יבגני-סיני",
    "einatwilf",
    "elianebarnett",
    "saurabh-kumar-82417a137",
    "david-osher",
    "nicoleeppolito",
    "ella-kenan",
    "michael-shafir",
    "blakejoubert",
    "shir-gross",
    "yair-berger-4b9911177",
    "alon-barak",
    "ravimishraphysics",
    "andresvourakis",
    "yarin-sultan",
    "gilad-tsehori",
    "benedictasare",
    "sapir-german-705ba41b3",
    "adi-ozer",
    "therohansheth",
    "mandowsky",
    "michael-kisilenko-ceo",
    "tal-hacmon1",
    "bareketmichaeli",
    "karin-hason-novo",
    "alon-reichman-4a765b213",
    "drlirazmargalit",
    "bar-rhamim",
    "roi-gerber",
    "allison-peck10000",
    "einat-sagee-alfasa-aa603820",
    "nivitzhaky",
    "rabbipoupko",
    "ronkonstantin",
    "rotembezalel",
    "racheal-kuranchie",
    "topaz-mothada",
    "adit-dan",
    "noa-hilzenrat",
    "adir-reuven"
]


def get_existing_users(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Convert the UserName column to a set for efficient lookups
    user_names_in_csv = set(df['UserName'])

    # Get only the users that are in the UserName column
    existing_users = [user for user in users if user in user_names_in_csv]

    return existing_users


def remove_duplicates(users_list):
    print(len(users_list))
    print(len(set(users_list)))
    return set(users_list)


def main():
    existing_users = get_existing_users('Linkedin_Posts_withGPT.csv')
    print("Users found in the CSV file:", existing_users)
    print("Num of Users found in the CSV file:", len(existing_users))
    print("Num of Users :", len(users))


if __name__ == '__main__':
    main()
    # print(remove_duplicates(users))
