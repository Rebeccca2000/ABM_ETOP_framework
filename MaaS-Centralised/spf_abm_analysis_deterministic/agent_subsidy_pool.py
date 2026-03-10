class SubsidyPoolConfig:
    def __init__(self, pool_type, amount):
        """
        Initialize subsidy pool configuration
        pool_type: 'daily', 'weekly', or 'monthly'
        amount: total subsidy amount for the period
        """
        self.pool_type = pool_type
        self.total_amount = amount
        
    def is_reset_time(self, current_step, last_reset_step):
        """
        Determine if it's time to reset the subsidy pool based on the pool type
        """
        steps_per_day = 144
        current_day = current_step // steps_per_day
        last_reset_day = last_reset_step // steps_per_day
        
        # Get day of week (0 = Monday, 6 = Sunday)
        day_of_week = current_day % 7
        
        if self.pool_type == 'daily':
            return current_day > last_reset_day
            
        elif self.pool_type == 'weekly':
            # Reset on Monday (day_of_week = 0)
            current_week = current_day // 7
            last_week = last_reset_day // 7
            return current_week > last_week and day_of_week == 0
            
        elif self.pool_type == 'monthly':
            # Assuming 30 days per month for simplicity
            current_month = current_day // 30
            last_month = last_reset_day // 30
            return current_month > last_month
            
        return False

    def is_subsidy_available(self, day_of_week):
        """
        Check if subsidies are available on the given day
        """
        if self.pool_type == 'weekly':
            # Only provide subsidies Monday through Friday (0-4)
            return day_of_week < 5
        return True
