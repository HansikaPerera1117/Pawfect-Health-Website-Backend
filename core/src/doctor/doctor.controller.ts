import { Body, Controller, Get, Param, Post } from '@nestjs/common';
import { DoctorService } from './doctor.service';

@Controller('doctor')
export class DoctorController {
  constructor(private readonly doctorService: DoctorService) {}

  @Post('create')
  create(@Body() data: any) {
    return this.doctorService.create(data);
  }

  @Post('find-nearest-doctor')
  findNearestDoctor(@Body() data: any) {
    return this.doctorService.findNearestDoctor(data);
  }

  @Get('find-by-id/' + ':doctorId')
  findById(@Param('doctorId') doctorId: string) {
    return this.doctorService.findById(doctorId);
  }

  @Post('login')
  login(@Body() data: any) {
    return this.doctorService.loginDoctor(data.email, data.password);
  }
  
}
