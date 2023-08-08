class LogColor:
	def __init__(self):
		self.HEADER    = '\033[95m'
		self.OKBLUE    = '\033[94m'
		self.OKCYAN    = '\033[96m'
		self.OKGREEN   = '\033[92m'
		self.WARNING   = '\033[93m'
		self.FAIL      = '\033[91m'
		self.ENDC      = '\033[0m'
		self.BOLD      = '\033[1m'
		self.UNDERLINE = '\033[4m'

	
	def my_print(self, text, color):
		'''
		HEADER    = '\033[95m'\n
		OKBLUE    = '\033[94m'\n
		OKCYAN    = '\033[96m'\n
		OKGREEN   = '\033[92m'\n
		WARNING   = '\033[93m'\n
		FAIL      = '\033[91m'\n
		ENDC      = '\033[0m'\n
		BOLD      = '\033[1m'\n
		UNDERLINE = '\033[4m'\n
		'''	
		print(f"{color}{text}{self.ENDC}")

	def p_header(self, text):
		print(f"{self.HEADER}{text}{self.ENDC}")

	def p_warn(self, text):
		print(f"[WARN]:\t{self.WARNING}{text}{self.ENDC}")

	def p_fail(self, text):
		print(f"[FAIL]:\t{self.FAIL}{text}{self.ENDC}")
	
	def p_ok(self, text):
		print(f"[OK]:\t{self.OKGREEN}{text}{self.ENDC}")
	
	def p_okblue(self, text):
		print(f"[OK]:\t{self.OKBLUE}{text}{self.ENDC}")
	
	def p_okcyan(self, text):
		print(f"[OK]:\t{self.OKCYAN}{text}{self.ENDC}")

	def p_bold(self, text):
		return f"{self.BOLD}{text}{self.ENDC}"
	
	def p_underline(self, text):
		return f"{self.UNDERLINE}{text}{self.ENDC}"
	

