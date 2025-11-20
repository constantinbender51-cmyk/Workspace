import datetime

def get_weekday():
    return datetime.datetime.now().strftime("%A")

def get_time():
    return datetime.datetime.now().strftime("%H:%M")

def get_uplifting_comment(weekday, time):
    hour = int(time.split(':')[0])
    
    if hour < 12:
        return f"Good morning! It's {weekday}. Start your day with positivity and purpose!"
    elif hour < 17:
        return f"Good afternoon! It's {weekday}. Keep up the great work and stay motivated!"
    elif hour < 21:
        return f"Good evening! It's {weekday}. You've accomplished so much today - be proud!"
    else:
        return f"Good night! It's {weekday}. Rest well and recharge for another amazing day!"

if __name__ == "__main__":
    weekday = get_weekday()
    current_time = get_time()
    comment = get_uplifting_comment(weekday, current_time)
    print(f"Current day: {weekday}")
    print(f"Current time: {current_time}")
    print(f"Uplifting comment: {comment}")