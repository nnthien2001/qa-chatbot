class IdGenerator:
  def __init__(self):
    self.current_id = 0
    
  def next(self):
    self.current_id += 1
    return self.current_id