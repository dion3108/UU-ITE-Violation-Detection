# pakai
import re
import string
import emoji
import unicodedata
import pandas as pd


# --- Fungsi bantu ---
def extract_links(text):
    return re.findall(r"https?://\S+", text)

# Mapping karakter non-Latin (Cyrillic) yang mirip huruf Latin
visual_map = {
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H',
    'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P', 'Т': 'T',
    'Х': 'X', 'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p',
    'с': 'c', 'у': 'y', 'х': 'x',
}

def merge_split_letters(text):
    """
    Gabungkan kembali huruf-huruf satuan yang dipisah spasi.
    Contoh: 'ᴡ ᴀ ᴋ ᴀ ᴛ ᴏ ᴛ ᴏ' → 'ᴡᴀᴋᴀᴛᴏᴛᴏ'
    """
    tokens = text.strip().split()
    merged = []
    temp = ""

    for token in tokens:
        if len(token) == 1 and unicodedata.category(token).startswith("L"):
            temp += token
        else:
            if temp:
                merged.append(temp)
                temp = ""
            merged.append(token)

    if temp:
        merged.append(temp)

    return " ".join(merged)


def preprocess_text(text):
    """
    Normalisasi teks komentar:
    - huruf kecil
    - link → LINK0, LINK1, ...
    - hapus timestamp (hh:mm)
    - hapus tanda baca ASCII (kecuali @ dan ?)
    - hapus karakter non-latin dan non-emoji dari tiap kata
    - rapikan spasi
    """
    text = str(text).lower()

    # Ganti semua link dengan token LINK0, LINK1, ...
    links = extract_links(text)
    for i, link in enumerate(links):
        text = text.replace(link, f"LINK{i}")

    # Hapus format jam (misalnya 11:35)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)

    # Hapus tanda baca ASCII (kecuali @ dan ?)
    ascii_punctuation = ''.join(c for c in string.punctuation if c not in "@?")
    text = re.sub(rf"[{re.escape(ascii_punctuation)}]", " ", text)

    # Ganti karakter non-printable
    text = re.sub(r"[\u0000-\u001F\u007F-\u009F\u200B-\u200D]+", " ", text)

    def is_valid_text_char(char):
        # PERBAIKAN: Memastikan return False berada di dalam blok try-except
        try:
            name = unicodedata.name(char)
            # Hanya huruf/angka Latin atau gaya Latin matematika (bold/italic/fancy)
            if "LATIN" in name or "MATHEMATICAL" in name or "DIGIT" in name:
                return True
        except ValueError:
            # Jika karakter tidak punya nama (misalnya, beberapa simbol), anggap tidak valid
            return False
        # Jika punya nama tapi tidak cocok kriteria di atas, anggap tidak valid
        return False

    def clean_word(word):
        if word.lower().startswith("link"):
            return word

        chars_to_remove = {'♥', '❤', '💖', '💘', '💗', '💓', '💝'}  # simbol love
        result = ""

        if word.startswith("@"):
            # Pecah dengan regex: @username + sisa karakter
            # Pecah @username dari sisanya
            match = re.match(r"^@[\w\d_]+", word) # match @username
            if match:
                username = match.group()
                tail = word[len(username):]
            else:
                username = "@"
                tail = word[1:]
            
            # Bersihkan tail: buang simbol non-emoji (kecuali emoji tetap)
            # Sisakan hanya emoji dari tail
            tail_cleaned = ''.join(
                c for c in tail if c not in chars_to_remove and c in emoji.EMOJI_DATA
            )

            return f"{username} {tail_cleaned}".strip()

        for char in word:
            if char in chars_to_remove:
                continue
            if char in visual_map:
                result += visual_map[char]
            elif is_valid_text_char(char) or char in "@?" or char in emoji.EMOJI_DATA:
                result += char

        return result





    words = text.strip().split()
    words = [clean_word(word) for word in words if clean_word(word)]
    text = " ".join(words)
    text = merge_split_letters(text)  # ⬅️ Tambahkan ini untuk gabungkan huruf visual
    
    # PERBAIKAN: Memastikan fungsi selalu mengembalikan string
    return text



def is_only_emoji(text):
    return all(char in emoji.EMOJI_DATA for char in text.strip()) and len(text.strip()) > 0

def is_only_numbers(text):
    return re.fullmatch(r"\d+", text.strip()) is not None

def is_only_mentions(text):
    return bool(re.fullmatch(r"(@[^\s@]+[\s]*)+", text.strip()))

def is_single_word(text):
    return len(text.strip().split()) == 1

def is_only_mentions_and_emoji(text):
    words = text.strip().split()
    if not words:
        return False
    for word in words:
        if word.startswith("@"):
            continue
        elif all(char in emoji.EMOJI_DATA for char in word):
            continue
        else:
            return False
    return True

def is_only_mentions_links_and_emoji(text):
    words = text.strip().split()
    if not words:
        return False
    for word in words:
        if word.startswith("@"):
            continue
        elif re.match(r"https?://\S+", word):
            continue
        elif word.lower().startswith("link"):
            continue
        elif all(char in emoji.EMOJI_DATA for char in word):
            continue
        else:
            return False
    return True

def is_only_symbols_or_emoticons(text):
    """
    Buang komentar jika:
    - Tidak mengandung huruf Latin (a-z)
    - Terdiri dari karakter simbol, bentuk dekoratif, atau emoticon teks
    """
    text = text.strip()
    if not text:
        return False

    # Jika ada huruf Latin, anggap ada kata bermakna
    if re.search(r"[a-z]", text):
        return False

    # Jika panjang teks minimal dan semua bukan angka/alfabet → buang
    if len(text) == 0: return False # Hindari ZeroDivisionError
    non_alnum_ratio = sum(1 for c in text if not c.isalnum()) / len(text)
    return non_alnum_ratio > 0.6  # 60% lebih bukan huruf/angka

def is_mention_and_short_word(text):
    """
    Buang komentar jika hanya terdiri dari mention + satu kata pendek (panjang ≤ 3 huruf).
    Contoh: "@3 am", "@akun wkwk"
    """
    tokens = text.strip().split()
    if len(tokens) == 2 and tokens[0].startswith("@"):
        word = tokens[1]
        return len(word) <= 5
    return False

# def is_dominantly_mention(text):
#     """
#     Jika 70% atau lebih token adalah mention (@...), buang.
#     Kecuali jika ada kalimat panjang setelahnya.
#     """
    
#     tokens = text.strip().split()
#     if not tokens:
#         return False
#     mention_count = sum(1 for token in tokens if token.startswith("@"))
#     ratio = mention_count / len(tokens)
#     return ratio >= 0.66 and len(tokens) <= 5
def is_dominantly_mention(text):
    """
    Buang komentar jika:
    - ≥ 2 mention
    - dan sisa token tidak mengandung kata panjang bermakna
    - dan total token ≤ 6
    """
    tokens = text.strip().split()
    if not tokens:
        return False

    mention_tokens = [t for t in tokens if t.startswith("@")]
    non_mentions = [t for t in tokens if not t.startswith("@")]

    if len(mention_tokens) < 2 or len(tokens) > 6:
        return False

    # Jika semua non-mention pendek (≤4) dan bukan emoji
    semua_pendek = all(
        len(t) <= 4 and not all(char in emoji.EMOJI_DATA for char in t)
        for t in non_mentions
    )

    return semua_pendek






def get_removal_reason(text):
    # PERBAIKAN: Menambahkan pengecekan tipe data untuk keamanan
    if not isinstance(text, str) or not text.strip():
        return "kosong_setelah_pembersihan"
    elif is_only_mentions_links_and_emoji(text):
        return "mention_dan_link_atau_emoji_saja"
    elif is_only_mentions_and_emoji(text):
        return "mention_dan_emoji_saja"
    elif is_only_emoji(text):
        return "emoji_saja"
    elif is_only_symbols_or_emoticons(text):
        return "simbol_non_teks"
    elif is_only_numbers(text):
        return "angka_saja"
    elif is_only_mentions(text):
        return "mention_saja"
    elif is_mention_and_short_word(text):
        return "mention_dan_kata_pendek"
    elif is_single_word(text):
        return "satu_kata_saja"
    elif is_dominantly_mention(text):
        return "mayoritas_mention"
    elif is_mention_and_short_word(text):
        return "mention_dan_kata_pendek"

    else:
        return None



# --- Komentar untuk diuji langsung ---
sample_comments = [
    "₍ ˃ᯅ˂ （ ͜•人 ͜•） • ‿ώ‿ ꪊꪻ",
    "mampir dong",
    "fyp gara² ͜•人 ͜•）",
    "₍ ˃ᯅ˂ （ ͜•人 ͜•） • ‿ώ‿ ꪊꪻ",
    "@˚✧ u t a r i ✧˚ @liga @cel",
    "@✨ @tino☄️🪐 ⚔️",
    "fyp gara² ͜•人 ͜•）",
    "@𝙈𝙖𝙥𝙡𝙖𝙮𝟕𝟖𝟗 💚congratulations💚",
    "pasti fyp karena ini ͜•人 ͜•）",
    "@macanhoki789꧂💛💜never give up brother💜💛",
    "gemes ͜•人 ͜•）",
    "𝗟𝗘𝗦𝗧𝗜𝟳𝟳💥 gacir, pertama kali awak jepe 10 juta langsung jozz! 🚀",
    "Baru coba ✌𝐌𝐀𝐍𝘿𝘼𝙇𝙄𝙆𝘼 𝟕𝟕✌, ternyata prosesnya beneran cepet!",
    "𝐆𝐒𝐎 𝟕𝟖𝟗 🥰💚 slalu baik bosku 💚🥰",
    "Sumpah penasaran, kok pada bahas 🐊M̲A̲N̲D̲A̲L̲I̲K̲A̲7̲7̲🐊 di kolom komentar?",
    "07:10 ✌ ABCD4D ✌ selalu memanjakan semua pemain !!",
    "Kaget sih awalnya aku kira bohongan,ternyata di ♥M♥A♥N♥D♥A♥L♥I♥K♥A♥7♥7♥ banyak banget benefitnya!.",
    "Udah paling top markotop cuman di █▓▒▒░░░MANDALIKA77░░░▒▒▓█.",
    "Yang butuh hiburan di ᆞMᆞᆞAᆞᆞNᆞᆞDᆞᆞAᆞᆞLᆞᆞIᆞᆞKᆞᆞAᆞᆞ7ᆞᆞ7ᆞ",
    "Cepet ajak temen kalian di M̳̿͟͞A̳̿͟͞N̳̿͟͞D̳̿͟͞A̳̿͟͞L̳̿͟͞I̳̿͟͞K̳̿͟͞A̳̿͟͞7̳̿͟͞7̳̿͟͞ bisa dapet saldo cuy",
    "Di 𝐷𝑂𝑅𝘈𝟳7 saya menemukan kesempatan, dari penjaja es kelapa jadi pemilik usaha makanan.",
    "Salut bang keren kontennya , salam Jepe DI ⭐𝗪𝗘𝗧𝗢𝗡𝟴𝟴",
    "🔥𝗕𝗔𝗡𝗝𝗔𝗥𝟰𝗗",
    "Update mbak4d2 terbaru hadir! deposit qris 1 detik langsung masuk !! keren sekarang!",
    "ini sih king aloy sama si 𝗕𝗥𝗢𝗪𝗜𝗡𝟰𝗗 sama sama kocak ya...mantap banget sihh di 𝗕𝗥𝗢𝗪𝗜𝗡𝟰𝗗 𝗕𝗢𝗡𝗨𝗦 𝟭𝟬𝟬% 𝗨𝗡𝗧𝗨𝗞 𝗡𝗘𝗪 𝗠𝗘𝗠𝗕𝗘𝗥 𝗕𝗘𝗕𝗔𝗦 𝗜𝗣 semoga makin jaya ya",
    "20:33 buset dah  ─═ 𝗕 𝗘 𝗥 𝗞 𝗔 𝗛 𝟵 𝟵 ═─",
    "YANG INI KAK PASTI WD , 𝘭𝘪𝘯𝘬𝘯𝘺𝘢 𝘣𝘪𝘴𝘢 𝘬𝘦𝘵𝘪𝘬 𝘥𝘪 𝘨𝘰𝘰𝘨𝘭𝘦 : 🔍 𝗧𝗛𝗢𝗥𝟯𝟭𝟭",
    "Iseng-iseng main А𝙀𝘙𝘖𝟪𝟾, malah jadi hobi, cuan terus!",
    "ᴡ ᴀ ᴋ ᴀ ᴛ ᴏ ᴛ ᴏ BAGI THR LEEE",
    "𝚁𝙼𝙰789 🌠⭐sehat selalu ketua🌠⭐7",
    "axl777 bangbang888 🥰",
    "AHMA𝘋𝙏O𝐓O aku cba, eh lngsg dapet cuan gede",
    "@pgc789🖤✈️ so happy king✈️✈️🏁🏁",
    "☯️🆔jp good ❤️pgc789 ❤️ happy always boskuh🇦🇱🇦🇱🇦🇱506",
    "🔥🐲🔥 anubis303 ❤️‍🔥jp mexwin❤️‍🔥 bahagia selalu ketua 🎊",
    "@ch4lox 11🪐 @m4k4n d4l4m🐉 @rkdkroken",
    "@struggle£ •",
    "@3 am",
    "@desawa @adeknya liam @change exe™",
    "@netral23 @tuan muda",
    "@★★★ @aditya saputra @okan1 @pemburu janda isi saldo dulu",
    "@bh 12 LINK0",
    "🥰✨🙏thankssss 🌸🀄️uburwiner🀄️🌸 bahagia selalu bossku 💮🉐035826",
    "@boyzxzy @라잘리✨ @ kenzchan ✓ @tokisaki kurumi🗿 @diyaa🤍 @slim shady2695 ehem ehem oto",
    "@s ᴋ ᴇ ᴘ ᴛ ɪ s @apisss @ibeng",
    "@@st• ar°ga",
    "@violettaメ♪♪ @? @may๑˙❥˙๑ @danzz hyper @s t i k m a n🦹 @afra @sara",
    "@tana ♥︎",
    "@lu² @ql6 @zynthiaè 🦋 @★ daayy 𐙚 @chaaa @giegie",
    "@giiee •ムイチロー @yoichi ಠ∀ಠ @ ☆naa aja☆ @mᥲᥣᥣᥕᥫ᭡ 🗿🗿",
]

# --- Proses dan tampilkan hasil ---
results = []
for comment in sample_comments:
    cleaned = preprocess_text(comment)
    reason = get_removal_reason(cleaned)
    status = "✅ Dipertahankan" if reason is None else f"🗑️ Dibuang ({reason})"
    results.append({
        "Asli": comment,
        "Bersih": cleaned,
        "Status": status
    })

# Tampilkan hasil dalam bentuk tabel
df = pd.DataFrame(results)
print(df.to_string(index=False))
