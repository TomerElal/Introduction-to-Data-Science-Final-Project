import pandas as pd

users = [
    "dvir-ross",
    "danielle-biton",
    "tali-libman-95781427",
    "ofir-zuk",
    "adit-dan",
    "erangefen",
    "oksana-matiiash-62542b95",
    "gilad-tsehori",
    "einat-sagee-alfasa-aa603820",
    "buchman",
    "jonathan-shavit-615bb0215",
    "bareketmichaeli",
    "amir-barkol",
    "na-ama-schultz-547027101",
    "noa-hilzenrat",
    "adi-ozer",
    "adir-reuven",
    "miriam-dissen",
    "roi-gerber",
    "yarin-sultan",
    "noatishby",
    "omri-madar",
    "lotem-aviv",
    "eitan-ben-avi-1b7899236",
    "naftalibennett",
    "yoseph-haddad-a97a47198",
    "joshagardner",
    "karin-hason-novo",
    "adi-soffer-teeni-3b754635",
    "yael-zeevi-kalomiti",
    "ella-kenan",
    "eriklidman",
    "einatwilf",
    "ofer-shaltiel-28423261",
    "shay-kallach-97a8b5146",
    "mirimendelson",
    "idit-bar-iditaravit-arab-business-culture",
    "eden-bibas",
    "hilla-regev",
    "shirbarnea",
    "noa-elharrar",
    "pavelberengoltz",
    "hilaoren1",
    "danysz",
    "mandowsky",
    "oren-helman-792657127",
    "elianebarnett",
    "drorgloberman",
    "david-osher",
    "stav-ziv",
    "michael-shafir",
    "merav-pirak",
    "tal-hacmon1",
    "allison-peck10000",
    "hasolidit",
    "danieldayan",
    "alon-reichman-4a765b213",
    "topaz-mothada",
    "dev-g-5949a031b",
    "saurabh-kumar-82417a137",
    "johnrushx",
    "jason-salt",
    "sudoyasir",
    "jamesfobrien",
    "rabbipoupko",
    "-amitcohen",
    "blaise-bevilacqua",
    "or-nisani-755930233",
    ]


def get_existing_users(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Convert the UserName column to a set for efficient lookups
    user_names_in_csv = set(df['UserName'])

    # Get only the users that are in the UserName column
    existing_users = [user for user in users if user in user_names_in_csv]

    return existing_users


def main():
    existing_users = get_existing_users('Linkedin_Posts.csv')
    print("Users found in the CSV file:", existing_users)
    print("Num of Users found in the CSV file:", len(existing_users))


if __name__ == '__main__':
    main()
