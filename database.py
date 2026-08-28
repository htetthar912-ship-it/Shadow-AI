import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# ==========================================
# ၁။ DATABASE SETUP & CONFIGURATION
# ==========================================
DATABASE_URL = "sqlite:///./shadow_ai.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ==========================================
# ၂။ DATABASE MODELS (သင့်ရဲ့ မူရင်း Tables များ)
# ==========================================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, default="")
    last_name = Column(String, default="")
    tier = Column(String, default="free")            # "free" | "premium"
    premium_expires_at = Column(String, default="")  # ISO date string, empty = not premium
    credit_balance = Column(Integer, default=0)
    is_admin = Column(Integer, default=0)             # 0 = User, 1 = Admin
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class UsageRecord(Base):
    __tablename__ = "usage_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    identity = Column(String, index=True, nullable=False)
    date = Column(String, index=True, nullable=False)
    text_count = Column(Integer, default=0)
    free_image_count = Column(Integer, default=0)


class CreditTransaction(Base):
    __tablename__ = "credit_transactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    identity = Column(String, index=True, nullable=False)
    amount = Column(Integer, nullable=False)   # positive = top-up, negative = spend
    reason = Column(String, default="")
    balance_after = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# Database Table များ စတင်ဆောက်လုပ်ရန် Function
def init_db():
    Base.metadata.create_all(bind=engine)

# Database Session ခေါ်ယူရန် Helper Function
def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise e


# ==========================================
# ၃။ USER & ADMIN MANAGEMENT FUNCTIONS
# ==========================================

# (က) အကောင့်အသစ်ဖွင့်ခြင်း (Default: Free User ဖြစ်သည်)
def register_user(email: str, first_name: str = "", last_name: str = ""):
    db = get_db()
    existing_user = db.query(User).filter(User.email == email).first()
    
    if existing_user:
        db.close()
        return f"အမှား - {email} ဖြင့် အကောင့်ဖွင့်ထားပြီးသား ဖြစ်ပါသည်။"

    new_user = User(
        email=email,
        first_name=first_name,
        last_name=last_name
    )
    db.add(new_user)
    db.commit()
    db.close()
    return f"အောင်မြင်သည် - {email} အတွက် အကောင့်အသစ် ဆောက်ပြီးပါပြီ။"


# (ခ) ပုံမှန်အကောင့်ကို Admin အဖြစ် ပြောင်းလဲခြင်း
def set_user_as_admin(email: str, status: int = 1):
    """status = 1 ဆိုလျှင် admin ခန့်မည်၊ status = 0 ဆိုလျှင် admin မှ ပြန်ဖြုတ်မည်"""
    db = get_db()
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        db.close()
        return "အမှား - User မတွေ့ရှိပါ။"
        
    user.is_admin = status
    db.commit()
    db.close()
    status_text = "Admin" if status == 1 else "Normal User"
    return f"အောင်မြင်သည် - {email} ကို {status_text} အဖြစ် ပြောင်းလဲပြီးပါပြီ။"


# ==========================================
# ၄။ PREMIUM & CREDIT MANAGEMENT FUNCTIONS
# ==========================================

# (ဂ) Premium User အဖြစ် သက်တမ်းသတ်မှတ်ပြီး အဆင့်တိုးမြှင့်ပေးခြင်း
def upgrade_to_premium(email: str, months: int = 1):
    db = get_db()
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        db.close()
        return "အမှား - User မတွေ့ရှိပါ။"
        
    user.tier = "premium"
    # လက်ရှိအချိန်ကနေ လအလိုက် ရက်ပေါင်းတွက်ချက်ပြီး သက်တမ်းကုန်ဆုံးမည့်ရက် သတ်မှတ်ခြင်း
    expiry_date = datetime.datetime.utcnow() + datetime.timedelta(days=30 * months)
    user.premium_expires_at = expiry_date.isoformat()
    
    db.commit()
    db.close()
    return f"အောင်မြင်သည် - {email} ကို {months} လစာ Premium အဖြစ် တိုးမြှင့်ပေးပြီးပါပြီ။ (သက်တမ်းကုန်ရက်: {user.premium_expires_at})"


# (ဃ) User ထံသို့ Credit ဝယ်ယူမှုအရ Credit ဖြည့်ပေးခြင်း
def add_user_credit(email: str, amount: int, reason: str = "Coin Purchased"):
    db = get_db()
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        db.close()
        return "အမှား - User မတွေ့ရှိပါ။"
        
    # User ရဲ့ credit ကို ပေါင်းထည့်သည်
    user.credit_balance += amount
    
    # သင့်ရဲ့ CreditTransaction Table မှာ မှတ်တမ်းအသစ်သိမ်းသည်
    trx = CreditTransaction(
        identity=email,
        amount=amount, # အပေါင်းကိန်း (ဝင်လာသော credit)
        reason=reason,
        balance_after=user.credit_balance
    )
    db.add(trx)
    db.commit()
    db.close()
    return f"အောင်မြင်သည် - {email} ထံသို့ {amount} credit ဖြည့်ပြီးပါပြီ။ လက်ရှိလက်ကျန်: {user.credit_balance}"


# ==========================================
# ၅။ SECURITY & CHECKING FUNCTIONS (အသုံးပြုခွင့် စစ်ဆေးရန်)
# ==========================================

# (င) Premium သက်တမ်း ကုန်/မကုန် စစ်ဆေးပေးမည့် Helper Function
def verify_and_get_user(email: str):
    """User ကို စစ်ဆေးပြီး Premium သက်တမ်းကုန်နေလျှင် 'free' သို့ အလိုအလျောက် ပြန်ချပေးမည်"""
    db = get_db()
    user = db.query(User).filter(User.email == email).first()
    
    if user and user.tier == "premium" and user.premium_expires_at:
        expiry = datetime.datetime.fromisoformat(user.premium_expires_at)
        # လက်ရှိအချိန်က သက်တမ်းကုန်ရက်ထက် ကျော်နေပါက
        if datetime.datetime.utcnow() > expiry:
            user.tier = "free"
            user.premium_expires_at = ""
            db.commit()
            
    return user # ပြင်ဆင်ပြီးသား user object ကို ပြန်ပေးမည်


# ==========================================
# ၆။ စမ်းသပ်အသုံးပြုနည်း နမူနာ (USAGE EXAMPLE)
# ==========================================
if __name__ == "__main__":
    # ပထမဆုံးအကြိမ် Database နဲ့ Table တွေ ဆောက်ရန် run ပါ
    init_db()
    print("Database initialised successfully.\n")
    
    test_email = "htetthar912@gmail.com"
    
    # ၁။ အကောင့်အသစ် ဆောက်ခြင်း
    print(register_user(test_email, "Htet", "Thar"))
    
    # ၂။ အကောင့်ကို Admin ခန့်ခြင်း
    print(set_user_as_admin(test_email, status=1))
    
    # ၃။ Premium ၁ လစာ ဝယ်ယူမှုကို ထည့်သွင်းခြင်း
    print(upgrade_to_premium(test_email, months=1))
    
    # ၄။ Credit ၁၀၀ ဖိုး ဝယ်ယူမှုကို ဖြည့်ပေးခြင်း
    print(add_user_credit(test_email, amount=100, reason="KPay ဖြင့် ဝယ်ယူမှု"))
    
    # ၅။ အသုံးပြုသူရဲ့ လက်ရှိ Status ကို ပြန်လည်စစ်ဆေးကြည့်ခြင်း
    current_user = verify_and_get_user(test_email)
    if current_user:
        print(f"\n--- {current_user.email} ၏ လက်ရှိအခြေအနေ ---")
        print(f"အမျိုးအစား (Tier): {current_user.tier}")
        print(f"Admin ဟုတ်မဟုတ်: {'ဟုတ်ပါသည်' if current_user.is_admin == 1 else 'မဟုတ်ပါ'}")
        print(f"လက်ကျန် Credit: {current_user.credit_balance}")
